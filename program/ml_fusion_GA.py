import os
import sys
import pickle
import pandas as pd
import numpy as np
import time
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
sys.path.append(os.path.abspath(''))
sys.path.append(os.path.abspath('..'))
from valid_metrices_p22 import *
from Genetic_algorithm import genetic_algorithm_ensemble, predict_with_gae

item_column =['Thre','Rec','Spe','Pre','Acc','MCC','F1','AUC','PRAUC'] 

def combine_model(train_data, valid_data, test_data, data_path, out_dir, kfold, columns):  
    prediction_result_cv = []
    prediction_result_test = []
    
    train_y, train_X = train_data[:,-1], train_data[:,:-1]
    valid_y, valid_X = valid_data[:,-1], valid_data[:,:-1]
    test_y, test_X = test_data[:,-1], test_data[:,:-1]
    
    os.makedirs('%s/%s/%s' % (data_path, out_dir, kfold), exist_ok=True)
    
    scores, best_weights = genetic_algorithm_ensemble(train_data, population_size=100000, num_generations=100, mutation_rate=0.5)              
    scores = scores.reshape(scores.shape[0], 1)
    #scores = rfc.predict_proba(valid_X)
    
    if os.path.isfile('%s/%s/%s/best_weights.pkl' % (data_path, out_dir, kfold)) :
        pass
    else:
       os.makedirs('%s/%s/%s' % (data_path, out_dir, kfold), exist_ok=True)
    pickle.dump(best_weights, open('%s/%s/%s/best_weights.pkl' % (data_path, out_dir, kfold), 'wb'))
    
    scores_cv = predict_with_gae(valid_data, best_weights)
    scores_cv = scores_cv.reshape(scores_cv.shape[0], 1)
    tmp_result = np.zeros((len(valid_y), 1+2))
    tmp_result[:, 0], tmp_result[:, 1:] = valid_y, scores_cv    
    prediction_result_cv.append(tmp_result)
             
    if test_X.shape[0] != 0:
        scores_test = predict_with_gae(test_data, best_weights)
        scores_test = scores_test.reshape(scores_test.shape[0], 1)
        tmp_result_test = np.zeros((len(test_y), 1+2))
        tmp_result_test[:, 0], tmp_result_test[:, 1:] = test_y, scores_test    
        prediction_result_test.append(tmp_result_test)           
        #print(prediction_result_cv[0]) #[1  0.094, 0.9057 ]
        
    valid_probs = prediction_result_cv[0][:,2]
    valid_labels = prediction_result_cv[0][:,0]    
    #print(np.array([valid_probs, valid_labels]).T)
    
    test_probs = prediction_result_test[0][:,2]
    test_labels = prediction_result_test[0][:,0]   

    cv_output = pd.DataFrame(np.array([valid_probs, valid_labels]).T,  columns=['prob', 'label'] )
    cv_output.to_csv('%s/%s/%s/val_roc.csv' % (data_path, out_dir, kfold))  
    
    df_valid = pd.DataFrame(valid_data, columns = columns)
    df_valid.to_csv('%s/%s/%s/val_roc_com.csv' % (data_path, out_dir, kfold))
     
    test_output = pd.DataFrame(np.array([test_probs, test_labels]).T,  columns=['prob', 'label'] )
    test_output.to_csv('%s/%s/%s/test_roc.csv' % (data_path, out_dir, kfold))
    
    #print(f'validation: prob label {valid_probs} {valid_labels} ')
    
    # metrics calculation
    th_, rec_, pre_, f1_, spe_, acc_, mcc_, auc_, pred_class, prauc_ = eval_metrics(valid_probs, valid_labels) 
    valid_matrices = th_, rec_, spe_, pre_,  acc_, mcc_, f1_, auc_, prauc_
    th_, rec_, pre_, f1_, spe_, acc_, mcc_, auc_, pred_class, prauc_ = th_eval_metrics(th_, test_probs, test_labels)
    test_matrices = th_, rec_, spe_, pre_,  acc_, mcc_, f1_, auc_, prauc_

    print_results(valid_matrices, test_matrices) 
    #print(f'valid_matrices {valid_matrices}')  
    
    df = pd.DataFrame([valid_matrices, test_matrices], index=['valid','test'], columns=item_column)
    df2 = pd.DataFrame([test_matrices], index=['test'], columns=item_column)
    
    return df, df2, best_weights


def train_test(kfold, data_path, out_dir, combination, columns):
    #feature combine for each fold
    train_data = []
    valid_data =[]
    test_data =[]
    for comb in combination:
        machine = comb[0]
        fea = comb[1] #encoding
        for datype in ['train', 'val','test']:
                fea_file = data_path + '/%s/%s/%s/%s_roc.csv' %(machine, fea, str(kfold), datype)
                fea_data = pd.read_csv(fea_file)
                if datype == 'train':
                    train_data.append(fea_data['prob'].values.tolist())
                    train_data.append(fea_data['label'].values.tolist())                
                elif datype =='val':
                    valid_data.append(fea_data['prob'].values.tolist())
                    valid_data.append(fea_data['label'].values.tolist())
                elif datype =='test':
                    test_data.append(fea_data['prob'].values.tolist())
                    test_data.append(fea_data['label'].values.tolist())
                else:
                    pass
    train_data = np.array(train_data).T                    
    valid_data = np.array(valid_data).T
    test_data = np.array(test_data).T    
    print(f'train_data\n {train_data.shape}')
    print(f'valid_data\n {valid_data.shape}')
    print(f'test_data\n {test_data.shape}')
    # Redundant labels [label,prob,label, prob,....] are removed
    train_data = np.delete(train_data, [i for i in range(1, 2*len(combination)-1, 2)], 1)
    valid_data = np.delete(valid_data, [i for i in range(1, 2*len(combination)-1, 2)], 1)
    test_data  = np.delete(test_data, [i for i in  range(1, 2*len(combination)-1, 2)], 1) 
    print(f'train_data\n {train_data.shape}')
    print(f'valid_data\n {valid_data.shape}')
    print(f'test_data\n {test_data.shape}')
 
    # training and testing
    df, df2, best_weights = combine_model(train_data, valid_data, test_data, data_path, out_dir, kfold, columns)
    
    return df, df2, best_weights
    

def ranking(measure_path, machine_method_1, encode_method_1, machine_method_2, encode_method_2):
    columns_measure= ['Machine','Encode','Threshold', 'Sensitivity', 'Specificity', 'Precision','Accuracy', 'MCC', 'F1', 'AUC', 'AUPRC']
    #print(f'encode_method {encode_method_1}')
    infile_name = ["val_measures.csv", "test_measures.csv" ]

    val_measure  = []
    for machine_method in machine_method_1:
        for i, encode_method in enumerate(encode_method_1):
            infile_path = measure_path + "/%s/%s" %(machine_method, encode_method)       
            infile1 = infile_path + '/' + infile_name[0] #val
            #print(encode_method)
            #print(infile1)
            val_measure.append( [machine_method, encode_method] +  (pd.read_csv(infile1, index_col=0).iloc[-1].values.tolist())) # means

    for machine_method in machine_method_2:
        for i, encode_method in enumerate(encode_method_2):
            infile_path = measure_path + "/%s/%s" %(machine_method, encode_method)
            infile1 = infile_path + '/' + infile_name[0] #val
            val_measure.append( [machine_method, encode_method] +  (pd.read_csv(infile1, index_col=0).iloc[-1].values.tolist())) # means

    df_val_measure  = pd.DataFrame(data=val_measure, columns=columns_measure)
    
    # sort
    df_val_measure_sort = df_val_measure.sort_values('AUC', ascending=False)   
    val_measure = df_val_measure_sort.values.tolist()
    #print(val_measure)
    df_val_measure_sort.to_csv('ranking.csv')
     
    combination=[]
    for line in val_measure:
        combination.append([line[0], line[1]])        
    
    return combination   

# score combine method based on logistic regression
if __name__ == '__main__':   

    parser = argparse.ArgumentParser()
    parser.add_argument('--machine_method_1', type=str, help='term')
    parser.add_argument('--encode_method_1', type=str, help='term')
    parser.add_argument('--machine_method_2', type=str, help='term')
    parser.add_argument('--encode_method_2', type=str, help='term')
    parser.add_argument('--top_list', type=str, help='term')
    parser.add_argument('--species', type=str, help='term')
    args = parser.parse_args()
    
    machine_method_1 = args.machine_method_1.strip().split()
    encode_method_1  = args.encode_method_1.strip().split()
    machine_method_2 = args.machine_method_2.strip().split()
    encode_method_2  = args.encode_method_2.strip().split() 
    
    species = args.species
    top_list  =  args.top_list.strip().split(',')
    top_list = [int(i) for i in top_list]
    kfold=10
    print(top_list)

    data_path = "../data/result_%s" %species 

    combination_rank = ranking(data_path, machine_method_1, encode_method_1, machine_method_2, encode_method_2)  
       
    df_all = pd.DataFrame(columns=item_column)     
    for top_number in top_list:
        print(f'top_number: {top_number}')

        comb = []
        for i in range(top_number):
            comb.append(combination_rank[i]) 
        combination = comb

        columns = []      
        columns = [combination[i][0]+'-'+combination[i][1] for i in range(0,top_number)] + ['label']
        top_combination = [ combination[i] for i in range(0,top_number)]
        out_dir ='combine_GA/top%s'%top_number
        
        print(top_combination )
        df_train = pd.DataFrame(columns=item_column) 
        df_valid = pd.DataFrame(columns=item_column) 
        df_test = pd.DataFrame(columns=item_column)     
        df_weight = pd.DataFrame(columns= [f'{combination[i][0]}-{combination[i][1]}' for i in range(len(top_combination))])
            
        for k in range(1, kfold+1):
            print(f'Fold_number: {k}')
            df, df2, weight = train_test(k, data_path, out_dir, top_combination, columns)      
            df_valid.loc[str(k) + "_valid"] = df.loc['valid']
            df_test.loc[str(k) + "_test"] = df.loc['test']          
            df_weight.loc[str(k) + "_weight"] = weight
    
        df_cat = pd.DataFrame(columns=item_column)
        df_cat.loc["mean_train"] = df_valid.mean()
        df_cat.loc["sd_train"] = df_valid.std()
        df_cat.loc["mean_test"] = df_test.mean()
        df_cat.loc["sd_test"] = df_test.std()

        df_weight.loc["mean_weight"] = df_weight.mean()
        df_weight.loc["sd_weight"] = df_weight.std()
        
        print(df_cat )       
        df_cat.to_csv('%s/%s/average_measure.csv' %(data_path, out_dir)) 
        df_weight.to_csv('%s/%s/ga_weight.csv' %(data_path, out_dir)) 
                  
        df_all.loc[str(top_number) + '_valid'] = df_valid.mean()
        df_all.loc[str(top_number) + '_test'] = df_test.mean()
        
        
    df_all.to_csv('%s/combine_GA/top_measure.csv' %(data_path))         
          
    



    
    
    
       
