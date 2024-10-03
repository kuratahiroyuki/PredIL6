import numpy as np
import torch
from sklearn.metrics import matthews_corrcoef, confusion_matrix, roc_curve, auc, precision_recall_curve
import prettytable as pt
import numbers


class AUCMeter():
    """
    The AUCMeter measures the area under the receiver-operating characteristic
    (ROC) curve for binary classification problems. The area under the curve (AUC)
    can be interpreted as the probability that, given a randomly selected positive
    example and a randomly selected negative example, the positive example is
    assigned a higher score by the classification model than the negative example.

    The AUCMeter is designed to operate on one-dimensional Tensors `output`
    and `target`, where (1) the `output` contains model output scores that ought to
    be higher when the model is more convinced that the example should be positively
    labeled, and smaller when the model believes the example should be negatively
    labeled (for instance, the output of a signoid function); and (2) the `target`
    contains only values 0 (for negative examples) and 1 (for positive examples).
    """

    def __init__(self):
        super(AUCMeter, self).__init__()
        self.reset()

    def reset(self):
        self.scores = torch.DoubleTensor(torch.DoubleStorage()).numpy()
        self.targets = torch.LongTensor(torch.LongStorage()).numpy()

    def add(self, output, target):
        if torch.is_tensor(output):
            output = output.cpu().squeeze().numpy()
        if torch.is_tensor(target):
            target = target.cpu().squeeze().numpy()
        elif isinstance(target, numbers.Number):
            target = np.asarray([target])
        assert np.ndim(output) == 1, \
            'wrong output size (1D expected)'
        assert np.ndim(target) == 1, \
            'wrong target size (1D expected)'
        assert output.shape[0] == target.shape[0], \
            'number of outputs and targets does not match'
        assert np.all(np.add(np.equal(target, 1), np.equal(target, 0))), \
            'targets should be binary (0, 1)'

        self.scores = np.append(self.scores, output)
        self.targets = np.append(self.targets, target)
        
        #return self.scores, self.targets

    def value(self):
        # case when number of elements added are 0
        if self.scores.shape[0] == 0:
            return 0.5

        # sorting the arrays
        scores, sortind = torch.sort(torch.from_numpy(
            self.scores), dim=0, descending=True)
        scores = scores.numpy()
        sortind = sortind.numpy()

        # creating the roc curve
        tpr = np.zeros(shape=(scores.size + 1), dtype=np.float64)
        fpr = np.zeros(shape=(scores.size + 1), dtype=np.float64)

        for i in range(1, scores.size + 1):
            if self.targets[sortind[i - 1]] == 1:
                tpr[i] = tpr[i - 1] + 1
                fpr[i] = fpr[i - 1]
            else:
                tpr[i] = tpr[i - 1]
                fpr[i] = fpr[i - 1] + 1

        tpr /= (self.targets.sum() * 1.0)
        fpr /= ((self.targets - 1.0).sum() * -1.0)

        # calculating area under curve using trapezoidal rule
        n = tpr.shape[0]
        h = fpr[1:n] - fpr[0:n - 1]
        sum_h = np.zeros(fpr.shape)
        sum_h[0:n - 1] = h
        sum_h[1:n] += h
        area = (sum_h * tpr).sum() / 2.0

        return (area, tpr, fpr)

    def auc(self):
        # case when number of elements added are 0
        if self.scores.shape[0] == 0:
            return 0.5
    
        precision, recall, thresholds_pr = precision_recall_curve(y_true=self.targets, probas_pred=self.scores)
        prauc = auc(x=recall, y=precision) 
        
        fpr, tpr, thresholds = roc_curve(y_true=self.targets, y_score=self.scores)
        rocauc = auc(x=fpr, y=tpr)        
        
        return (rocauc, prauc)


def eval_metrics(probs, targets, cal_AUC=True):

    threshold_list = []
    for i in range(1, 50):
        threshold_list.append(i / 50.0)

    if cal_AUC:
        if isinstance(probs, torch.Tensor) and isinstance(targets, torch.Tensor):
            fpr, tpr, thresholds = roc_curve(y_true=targets.detach().cpu().numpy(), y_score=probs.detach().cpu().numpy())

            precision, recall, thresholds_pr = precision_recall_curve(y_true=targets.detach().cpu().numpy(), probas_pred=probs.detach().cpu().numpy())
            
        elif isinstance(probs, np.ndarray) and isinstance(targets, np.ndarray):
            fpr, tpr, thresholds = roc_curve(y_true=targets,y_score=probs)
            precision, recall, thresholds_pr = precision_recall_curve(y_true=targets, probas_pred=probs)
        else:
            print('ERROR: probs or targets type is error.')
            raise TypeError
        auc_ = auc(x=fpr, y=tpr)
        prauc_ = auc(x=recall, y=precision)
    else:
        auc_ = 0
        prauc_ = 0

    threshold_best, rec_best, pre_best, F1_best, spe_best, acc_best, mcc_best, pred_bi_best = 0, 0, 0, 0, 0, 0, -1, None
    for threshold in threshold_list:
        threshold, rec, pre,F1, spe, acc, mcc, _, pred_bi, _ = th_eval_metrics(threshold, probs, targets,cal_AUC=False)
        if mcc > mcc_best:
            threshold_best, rec_best, pre_best, F1_best, spe_best, acc_best, mcc_best, pred_bi_best = threshold, rec, pre, F1, spe, acc, mcc, pred_bi

    return threshold_best, rec_best, pre_best, F1_best, spe_best, acc_best, mcc_best, auc_, pred_bi_best, prauc_

def th_eval_metrics(threshold, probs, targets, cal_AUC=True):
    if isinstance(probs, torch.Tensor) and isinstance(targets,torch.Tensor):
        if cal_AUC:
            fpr, tpr, thresholds = roc_curve(y_true=targets.detach().cpu().numpy(), y_score=probs.detach().cpu().numpy())
            auc_ = auc(x=fpr, y=tpr)
            precision, recall, thresholds_pr = precision_recall_curve(y_true=targets.detach().cpu().numpy(), probas_pred=probs.detach().cpu().numpy())
            prauc_ = auc(x=recall, y=precision)
        else:
            auc_ = 0
            prauc_ = 0
        pred_bi = targets.data.new(probs.shape).fill_(0)
        pred_bi[probs>threshold] = 1 ###
        targets[targets==0] = 5
        targets[targets==1] = 10
        tn = torch.where((pred_bi+targets)==5)[0].shape[0]
        fp = torch.where((pred_bi+targets)==6)[0].shape[0]
        fn = torch.where((pred_bi+targets)==10)[0].shape[0]
        tp = torch.where((pred_bi+targets)==11)[0].shape[0]
        if tp>0:
            rec = tp / (tp + fn)
        else:
            rec = 0
        if tp > 0:
            pre = tp / (tp + fp)
        else:
            pre = 0
        if tn > 0:
            spe = tn / (tn + fp)
        else:
            spe = 0
        if (tn + tp) > 0:
            acc = (tn + tp) / (tn + fp + tp + fn)
        else:
            acc = 0
        if rec+pre > 0:
            F1 = 2 * rec * pre / (rec + pre)
        else:
            F1 = 0
        mcc = (tp*tn-fp*fn)/torch.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))

    elif isinstance(probs, np.ndarray) and isinstance(targets, np.ndarray):
        fpr, tpr, thresholds = roc_curve(y_true=targets, y_score=probs)
        auc_ = auc(x=fpr, y=tpr)
        precision, recall, thresholds_pr = precision_recall_curve(y_true=targets, probas_pred=probs)
        prauc_ = auc(x=recall, y=precision) 

        pred_bi = np.abs(np.ceil(probs - threshold)) ###

        tn, fp, fn, tp = confusion_matrix(targets, pred_bi).ravel()
        if tp >0 :
            rec = tp / (tp + fn)
        else:
            rec = 1e-8
        if tp >0:
            pre = tp / (tp + fp)
        else:
            pre = 1e-8
        if tn >0:
            spe = tn / (tn + fp)
        else:
            spe = 1e-8
        if (tn + tp) >0:
            acc = (tn + tp) / (tn + fp + tp + fn)
        else:
            acc = 1e-8
        #mcc = matthews_corrcoef(targets, pred_bi)
        if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) > 0:
            mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        else:
            mcc = -1
        if rec + pre > 0:
            F1 = 2 * rec * pre / (rec + pre)
        else:
            F1 = 0
    else:
        print('ERROR: probs or targets type is error.')
        raise TypeError

    return threshold, rec, pre, F1, spe, acc, mcc, auc_, pred_bi, prauc_

def CFM_eval_metrics(CFM):
    CFM = CFM.astype(float)
    tn = CFM[0, 0]
    fp = CFM[0, 1]
    fn = CFM[1, 0]
    tp = CFM[1, 1]
    if tp > 0:
        rec = tp / (tp + fn)
    else:
        rec = 0
    if tp > 0:
        pre = tp / (tp + fp)
    else:
        pre = 0
    if tn > 0:
        spe = tn / (tn + fp)
    else:
        spe = 0
    if (tn + tp) > 0:
        acc = (tn + tp) / (tn + fp + tp + fn)
    else:
        acc = 0
    if rec + pre > 0:
        F1 = 2 * rec * pre / (rec + pre)
    else:
        F1 = 0
    if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) > 0:
        mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    else:
        mcc = -1
    return rec, pre, F1, spe, acc, mcc

def print_results(valid_matrices=None,test_matrices=None):

    tb = pt.PrettyTable()
    tb.field_names = ['Dataset','th','Rec', 'Spe', 'Pre', 'Acc', 'MCC', 'F1', 'AUC', 'PRAUC']


    if valid_matrices is not None:
        row_list = ['valid']
        for i in range(9):
            row_list.append('{:.3f}'.format(valid_matrices[i]))
        tb.add_row(row_list)

    if test_matrices is not None:
        row_list = ['test']
        for i in range(9):
            row_list.append('{:.3f}'.format(test_matrices[i]))
        tb.add_row(row_list)
    print(tb)


