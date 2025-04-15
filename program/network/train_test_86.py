#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""
import sys
import os
import pandas as pd
import torch
import torch.nn.utils.rnn as rnn
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import DataLoader
from torch.autograd import Variable
#import torch_optimizer as optim
from torch import optim
from torch.utils.data import BatchSampler
import numpy as np
from numpy import argmax
import joblib
import argparse
from gensim.models import KeyedVectors
from gensim.models import word2vec
import sentencepiece as spm
import copy
import json
from TX_network import TX
from LSTM_network import Lstm
#from GRU_network_bidirectional import bGRU
from LSTM_network_bidirectional import bLSTM
#from CNN_LSTM_network import CNN_LSTM  #train_test_87.py
from CNN_network import CNN
import collections
import time
import pickle
import sklearn.metrics as metrics
from sklearn.metrics import precision_recall_curve
from loss_func import CBLoss
from metrics import cofusion_matrix, sensitivity, specificity, auc, mcc, accuracy, precision, recall, f1, cutoff, AUPRC
from sklearn.model_selection import StratifiedKFold
#metrics_dict = {"sensitivity":sensitivity, "specificity":specificity, "accuracy":accuracy,"mcc":mcc,"auc":auc,"precision":precision,"recall":recall,"f1":f1,"AUPRC":AUPRC}
metrics_dict = {"sensitivity":sensitivity, "specificity":specificity, "accuracy":accuracy,"auc":auc,"recall":recall,"f1":f1,"AUPRC":AUPRC}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#ESM-2
aa2index_esm_dict = {'A': 5, 'R': 10, 'N': 17, 'D': 13, 'C': 23, 'E': 9, 'Q': 16, 'G': 6, 'H': 21, 'I': 12, 'L': 4, 'K': 15, 'M': 20, 'F': 18, 'P': 14, 'S': 8, 'T': 11, 'W': 22, 'Y': 19, 'V': 7, 'X': 1, 'B': 25, 'O': 28, 'U': 26, 'Z': 27, '-': 1}
VOCAB_SIZE=25

def load_json(filename):
    with open(filename) as f:
        data = json.load(f)     
    return data

def file_input_csv(filename, index_col = None):
    data = pd.read_csv(filename, index_col = index_col)
    return data

def trim_input_csv(filename, seqwin, index_col = None):
    df1 = pd.read_csv(filename, index_col = index_col)
    seq = df1.loc[:,'seq'].tolist()
    #data triming
    for i in range(len(seq)):
        if len(seq) > seqwin:
            seq[i] = seq[i][0:seqwin]
    for i in range(len(seq)):
        df1.loc[i,'seq'] = seq[i]   
    return df1
        
def aa2be_dict_construction():
   AA = 'ARNDCQEGHILKMFPSTWYVBJOZX'
   keys=[]
   vectors=[]
   for i, key in enumerate(AA) :
      base=np.zeros(25)
      keys.append(key)
      base[i]=1
      vectors.append(base)
   aa2be_dict = dict(zip(keys, vectors))
   return aa2be_dict

def seq2esm2_dict_construction(whole_seq, esm2_model, aa2index, seqwin):    
    esm2_model.eval()  # disables dropout for deterministic results    
    tokens = []
    for seq in whole_seq:
        tok =[0]
        seq_len = len(seq)
        for aa in seq:
            tok += [aa2index[aa]]
        if seq_len <= seqwin:
            tok += [2]
            for i in range(seqwin-seq_len):
                tok += [1] 
        tokens.append(tok)
    tokens = torch.tensor(tokens)

    with torch.no_grad():
        results = esm2_model(tokens, repr_layers=[33], return_contacts=True)
    token_representations = results["representations"][33] # seqwin+2
    print(f'token_representations: {token_representations.shape}') #torch.Size([341, 52, 1280])  
    
    seq2esm2_dict = {}  
    for i in range(token_representations.shape[0]):
        seq2esm2_dict[whole_seq[i]] = token_representations[i,:,:].squeeze(0)
        
    return seq2esm2_dict
   
def emb_seq_BE(seq, aa2be_dict, num, seqwin, features):
    seq_emb = []
    for i in range(len(seq) - num + 1):
        try:
            x = aa2be_dict[seq[i]]
        except:
            x = np.zeros(features)
            print("key error in BE")
        seq_emb.append(x)
    seq_emb=np.array(seq_emb)

    pad = np.zeros((seqwin-len(seq), features))
    seq_emb = np.concatenate([seq_emb, pad])
        
    return seq_emb

def emb_seq_NN(seq, nn_embed, aa2index_esm_dict, seqwin, features):
    seq_emb = []

    for i in range(len(seq)):
        try:
            x = aa2index_esm_dict[seq[i]]
        except:
            x = 1
            print("key error in NN")
        seq_emb.append(x)
    seq_emb=torch.tensor(seq_emb, dtype=torch.int32)
    pad = torch.ones(seqwin-len(seq), dtype=torch.int32)
    seq_emb = torch.cat([seq_emb, pad])

    #nn_embed = nn.Embedding(VOCAB_SIZE, features)
    seq_emb_2 = nn_embed(seq_emb)
    #print(seq_emb_2)
    mask = torch.cat([torch.ones(len(seq)), torch.zeros(seqwin-len(seq))]) #[len, 64]
    #seq_emb_2 = seq_emb_2*mask[:,None] #masking greatly decrease the prediction

    return seq_emb_2
  
def emb_seq_w2v(seq, w2v_model, num, seqwin, features):
    seq_emb = []
    for i in range(len(seq) - num + 1):
        try:
            x = w2v_model.wv[seq[i:i+num]]
        except:
            x = np.zeros(features)
            #print("key error in W2V")
        seq_emb.append(x)
    pad = np.zeros((seqwin-len(seq), features))
    seq_emb = np.concatenate([seq_emb, pad]) 

    return seq_emb

def emb_seq_esm2(seq, esm2_model, seqwin, features) : 
    esm2_model.eval()
    
    tokens = [0]
    for aa in seq:
        tokens += [aa2index_esm_dict[aa]]
    tokens += [2]
    for i in range(seqwin-len(seq)):
        tokens += [1] 
    
    #print(tokens)   
    tokens = torch.tensor(tokens).unsqueeze(0) #torch.Size([seq_len +2, 1280])
    #print(tokens.shape)

    with torch.no_grad():
        results = esm2_model(tokens, repr_layers=[33], return_contacts=True)
    seq_emb = results["representations"][33].squeeze(0) # seqwin+2

    return seq_emb
   

def emb_seq_w2v_bpe(seq, w2v_model, num, seqwin, features):
    if num != 1:
        print("W2V kmer error")
        exit()
    seq_emb = []
    for i in range(len(seq) - num + 1):
        try:
            x = w2v_model.wv[seq[i]]
        except:
            x = np.zeros(features)
            print(f"key error in W2V {seq[i]}")
        seq_emb.append(x)
    pad = np.zeros((seqwin-len(seq), features))
    seq_emb = np.concatenate([seq_emb, pad]) 

    return seq_emb


class pv_data_sets():
    def __init__(self, data_sets, encode_method, aa2be_dict, kmer, w2v_model, features, sp, esm2_model, seq2esm2_dict, nn_embed, seqwin):
        super().__init__()
        self.seq = data_sets["seq"].values.tolist() #"sequence"
        self.labels = np.array(data_sets["label"].values.tolist()).reshape([len(data_sets["label"].values.tolist()),1]).astype(np.float32)
        self.encode_method = encode_method
        self.aa2be_dict = aa2be_dict
        self.seq2esm2_dict = seq2esm2_dict
        self.kmer = kmer
        self.w2v_model = w2v_model
        self.nn_embed = nn_embed
        self.features=features
        self.sp = sp
        self.esm2 = esm2_model
        self.seqwin = seqwin

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        label = self.labels[idx] 
        if self.encode_method == 'BE':
            emb_mat = emb_seq_BE(self.seq[idx], self.aa2be_dict, self.kmer, self.seqwin, self.features)
        elif self.encode_method == 'W2V':
            emb_mat = emb_seq_w2v(self.seq[idx], self.w2v_model, self.kmer, self.seqwin, self.features)
        elif self.encode_method == 'W2V_BPE':
            subword_seq = self.sp.EncodeAsPieces(self.seq[idx])
            #print(subword_seq) #['HVR', 'HFY', 'GL', 'M']
            emb_mat = emb_seq_w2v_bpe(subword_seq, self.w2v_model, 1, self.seqwin, self.features)
        elif self.encode_method == 'NN':
            emb_mat = emb_seq_NN(self.seq[idx], self.nn_embed, aa2index_esm_dict, self.seqwin, self.features)
            #print(f'emb_mat: {emb_mat.shape}')
            return emb_mat.float().to(device), torch.tensor(label).to(device)
        elif self.encode_method == 'ESM2':
            emb_mat = self.seq2esm2_dict[self.seq[idx]][...,:self.features]
            return emb_mat.float().to(device), torch.tensor(label).to(device)
        else:
            print('no encoding method')
            exit()
              
        return torch.tensor(emb_mat).float().to(device), torch.tensor(label).to(device)

class train_test_process():
    def __init__(self, out_path, loss_type = "balanced", tra_batch_size = 128, val_batch_size = 128, test_batch_size = 32, lr = 0.001, n_epoch = 10000, early_stop = 25, thresh = 0.5): #lr = 0.00001,
        self.out_path = out_path
        self.tra_batch_size = tra_batch_size
        self.val_batch_size = val_batch_size

        self.lr = lr
        self.n_epoch = n_epoch
        self.early_stop = early_stop
        self.thresh = thresh
        self.loss_type = loss_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def training_testing(self, train_data_sets, val_data_sets, test_data_sets, deep_method, encode_method, seqwin, kmer, w2v_model, vector_size, sp_bpe, esm2_model, seq2esm2_dict):
        os.makedirs(self.out_path + "/data_model", exist_ok=True)
        
        nn_embed =None
        aa2be_dict = aa2be_dict_construction()
        if encode_method == 'W2V':
            features = vector_size
        elif encode_method == 'W2V_BPE':
            features = vector_size
        elif encode_method == 'NN':          
            features = vector_size
            nn_embed = nn.Embedding(VOCAB_SIZE, features)
        elif encode_method == 'ESM2':
            features = vector_size   #1280    
        elif encode_method == 'BE':
            features = 25*kmer
        else:
            
            print("No encoding method")
       
        tra_data_all = pv_data_sets(train_data_sets, encode_method, aa2be_dict, kmer, w2v_model, features, sp_bpe, esm2_model, seq2esm2_dict, nn_embed, seqwin)
        train_loader = DataLoader(dataset = tra_data_all, batch_size = self.tra_batch_size, shuffle=False)

        val_data_all = pv_data_sets(val_data_sets, encode_method, aa2be_dict, kmer, w2v_model, features, sp_bpe, esm2_model, seq2esm2_dict, nn_embed, seqwin)
        val_loader = DataLoader(dataset = val_data_all, batch_size = self.val_batch_size, shuffle=False)
        
        test_data_all = pv_data_sets(test_data_sets, encode_method, aa2be_dict, kmer, w2v_model, features, sp_bpe, esm2_model, seq2esm2_dict, nn_embed, seqwin)
        test_loader = DataLoader(dataset = test_data_all, batch_size = 32, shuffle=False) 
                       
        if deep_method == 'CNN':
            if encode_method == 'ESM2':
                net = CNN(features = features, time_size = seqwin - kmer + 3).to(device) # seq の両端に先頭、後尾の表す文字を入っているので
            else :
                net = CNN(features = features, time_size = seqwin - kmer + 1).to(device)       
        elif deep_method == 'LSTM':  
            net = Lstm(features = features, lstm_hidden_size = 128).to(device)
        elif deep_method == 'bLSTM': 
            net = bLSTM(features = features, lstm_hidden_size = 128).to(device)
        elif deep_method == 'TX' :
            if encode_method == 'ESM2':   
                net = TX(n_layers=3, d_model=features, n_heads=4, d_dim=100, d_ff=400, time_seq=seqwin-kmer+3).to(device) 
            else :
                net = TX(n_layers=3, d_model=features, n_heads=4, d_dim=100, d_ff=400, time_seq=seqwin-kmer+1).to(device)                         
        else:
           print('no net exist')
           exit()
               
        opt = optim.Adam(params = net.parameters(), lr = self.lr)
         
        if(self.loss_type == "balanced"):
            criterion = nn.BCELoss()
            
        min_loss = 1000
        early_stop_count = 0
        with open(self.out_path + "/cv_result.txt", 'w') as f:
            print(self.out_path, file = f, flush=True)
            print("The number of training data:" + str(len(train_data_sets)), file = f, flush=True)
            print("The number of validation data:" + str(len(val_data_sets)), file = f, flush=True)
                      
            for epoch in range(self.n_epoch):
                train_losses, val_losses = [], []
                self.train_probs, self.train_labels = [], []
                           
                print("epoch_" + str(epoch + 1) + "=====================", file = f, flush=True) 
                print("train...", file = f, flush=True)
                net.train()
                
                for i, (emb_mat, label) in enumerate(train_loader):
                    #print(f'emb_mat.shape {emb_mat.shape}') #torch.Size([128, 40, 64])
                    
                    opt.zero_grad()
                    outputs = net(emb_mat)
                    
                    if(self.loss_type == "balanced"):
                        loss = criterion(outputs, label)
                    elif(self.loss_type == "imbalanced"):
                        loss = CBLoss(label, outputs, 0.9999, 2)
                    else:
                        print("ERROR::You can not specify the loss type.")

                    loss.backward()
                    opt.step()
                    
                    train_losses.append(float(loss.item()))
                    self.train_probs.extend(outputs.cpu().clone().detach().squeeze(1).numpy().flatten().tolist())
                    self.train_labels.extend(label.cpu().clone().detach().squeeze(1).numpy().flatten().tolist())

                train_thresh = 0.5
                print("train_loss:: value: %f, epoch: %d" % (sum(train_losses) / len(train_losses), epoch + 1), file = f, flush=True) 
                print("train_loss:: value: %f, epoch: %d, time: %f" % (sum(train_losses) / len(train_losses), epoch + 1, time.time()-start)) 
                print("val_threshold:: value: %f, epoch: %d" % (train_thresh, epoch + 1), file = f, flush=True)
                for key in metrics_dict.keys():
                    if(key != "auc" and key != "AUPRC"):
                        metrics = metrics_dict[key](self.train_labels, self.train_probs, thresh = train_thresh)
                    else:
                        metrics = metrics_dict[key](self.train_labels, self.train_probs)
                    print("train_" + key + ": " + str(metrics), file = f, flush=True)
                    
                tn_t, fp_t, fn_t, tp_t = cofusion_matrix(self.train_labels, self.train_probs, thresh = train_thresh)
                print("train_true_negative:: value: %f, epoch: %d" % (tn_t, epoch + 1), file = f, flush=True)
                print("train_false_positive:: value: %f, epoch: %d" % (fp_t, epoch + 1), file = f, flush=True)
                print("train_false_negative:: value: %f, epoch: %d" % (fn_t, epoch + 1), file = f, flush=True)
                print("train_true_positive:: value: %f, epoch: %d" % (tp_t, epoch + 1), file = f, flush=True)

                print("validation...", file = f, flush=True)
                
                net.eval()
                self.val_probs, self.val_labels = [], []
                for i, (emb_mat, label) in enumerate(val_loader):
                    with torch.no_grad():
                        outputs = net(emb_mat)
                        
                    if(self.loss_type == "balanced"):
                        loss = criterion(outputs, label)
                    elif(self.loss_type == "imbalanced"):
                        loss = CBLoss(label, outputs, 0.9999, 2)
                    else:
                        print("ERROR::You can not specify the loss type.")

                    if(np.isnan(loss.item()) == False):
                        val_losses.append(float(loss.item()))
                        
                    self.val_probs.extend(outputs.cpu().detach().squeeze(1).numpy().flatten().tolist())
                    self.val_labels.extend(label.cpu().detach().squeeze(1).numpy().flatten().tolist()) 
                
                loss_epoch = sum(val_losses) / len(val_losses)

                val_thresh = 0.5

                print("validation_loss:: value: %f, epoch: %d" % (loss_epoch, epoch + 1), file = f, flush=True)
                print("val_threshold:: value: %f, epoch: %d" % (val_thresh, epoch + 1), file = f, flush=True)
                for key in metrics_dict.keys():
                    if(key != "auc" and key != "AUPRC"):
                        metrics = metrics_dict[key](self.val_labels, self.val_probs, thresh = val_thresh)
                    else:
                        metrics = metrics_dict[key](self.val_labels, self.val_probs)
                    print("validation_" + key + ": " + str(metrics), file = f, flush=True)
                
                tn_t, fp_t, fn_t, tp_t = cofusion_matrix(self.val_labels, self.val_probs, thresh = val_thresh)
                print("validation_true_negative:: value: %f, epoch: %d" % (tn_t, epoch + 1), file = f, flush=True)
                print("validation_false_positive:: value: %f, epoch: %d" % (fp_t, epoch + 1), file = f, flush=True)
                print("validation_false_negative:: value: %f, epoch: %d" % (fn_t, epoch + 1), file = f, flush=True)
                print("validation_true_positive:: value: %f, epoch: %d" % (tp_t, epoch + 1), file = f, flush=True)

                if loss_epoch < min_loss:
                    early_stop_count = 0
                    min_loss = loss_epoch
                    os.makedirs(self.out_path + "/data_model", exist_ok=True)
                    os.chdir(self.out_path + "/data_model")
                    torch.save(net.state_dict(), "deep_model")

                    final_thresh = 0.5
                    final_val_probs = self.val_probs  
                    final_val_labels = self.val_labels
                    final_train_probs = self.train_probs
                    final_train_labels = self.train_labels
                    
                else:
                    early_stop_count += 1
                    if early_stop_count >= self.early_stop:
                        print('Traning can not improve from epoch {}\tBest loss: {}'.format(epoch + 1 - self.early_stop, min_loss), file = f, flush=True)
                        break # Simulation continues until the end of epochs
                    
            #print(val_thresh, file = f, flush=True)
            for key in metrics_dict.keys():
                if(key != "auc" and key != "AUPRC"):
                    train_metrics = metrics_dict[key](final_train_labels,final_train_probs,thresh = final_thresh)
                    val_metrics = metrics_dict[key](final_val_labels,final_val_probs, thresh = final_thresh)
                else:
                    train_metrics = metrics_dict[key](final_train_labels, final_train_probs)
                    val_metrics = metrics_dict[key](final_val_labels, final_val_probs)
                print("train_" + key + ": " + str(train_metrics), file = f, flush=True)
                print("val_" + key + ": " + str(val_metrics), file = f, flush=True)
 
 
        ### testing process        
        with open(self.out_path + "/test_result.txt", 'w') as f:  
            print(self.out_path, file = f, flush=True)
            print("The number of testing data:" + str(len(test_data_sets)), file = f, flush=True)
            
            self.test_probs, self.test_labels = [], []
            
            print("testing...", file = f, flush=True)
            net.eval()

            
            for i, (emb_mat, label) in enumerate(test_loader):
                with torch.no_grad():                  
                    outputs = net(emb_mat) # deep learning model 1個ずつ処理できるのか？
                        
                self.test_probs.extend(outputs.cpu().detach().squeeze(1).numpy().flatten().tolist())
                self.test_labels.extend(label.cpu().detach().squeeze(1).numpy().flatten().tolist()) 
                
            #print("test_threshold:: value: %f" % (str(self.thresh)), file = f, flush=True)
            for key in metrics_dict.keys():
                if(key != "auc" and key != "AUPRC"):
                    test_metrics = metrics_dict[key](self.test_labels, self.test_probs, thresh = self.thresh)
                else:
                    test_metrics = metrics_dict[key](self.test_labels, self.test_probs)
                print("test_" + key + ": " + str(test_metrics), file = f, flush=True)
                
            tn_t, fp_t, fn_t, tp_t = cofusion_matrix(self.test_labels, self.test_probs, thresh = self.thresh)
            print("test_true_negative:: value: %f" % (tn_t), file = f, flush=True)
            print("test_false_positive:: value: %f" % (fp_t), file = f, flush=True)
            print("test_false_negative:: value: %f" % (fn_t), file = f, flush=True)
            print("test_true_positive:: value: %f" % (tp_t), file = f, flush=True)


###############################################################################################################              
if __name__ == '__main__':
    start=time.time()
              
    parser = argparse.ArgumentParser()
    parser.add_argument('--intrain', help='Path')
    parser.add_argument('--intest', help='Path')
    parser.add_argument('--outpath', help='Path')
    parser.add_argument('--losstype', help='Path', default = "balanced", choices=["balanced", "imbalanced"])
    parser.add_argument('--w2vmodel', help='Path')
    parser.add_argument('--deeplearn', help='Path')
    parser.add_argument('--encode', help='Path')
    parser.add_argument('--kfold', type=int, help='Path')
    parser.add_argument('--seqwin', type=int, help='Path')
    parser.add_argument('--kmer', type=int, help='Path')
    parser.add_argument('--size', type=int, help='Path')
    parser.add_argument('--epochs', type=int, help='Path')
    parser.add_argument('--sg', type=int, help='Path')
    parser.add_argument('--window', type=int, help='Path')
    parser.add_argument('--w2v_bpe_model', help='Path') 
    parser.add_argument('--bpe_model', help='Path') 
    parser.add_argument('--esm2', default=None, help='Path') 
  
    path = parser.parse_args().intrain
    test_path = parser.parse_args().intest
    out_path = parser.parse_args().outpath
    loss_type = parser.parse_args().losstype
    w2v_model = parser.parse_args().w2vmodel
    deep_method = parser.parse_args().deeplearn
    encode_method = parser.parse_args().encode
    kfold = parser.parse_args().kfold
    seqwin = parser.parse_args().seqwin
    kmer = parser.parse_args().kmer
    size = parser.parse_args().size
    epochs=parser.parse_args().epochs
    sg = parser.parse_args().sg
    window = parser.parse_args().window

    w2v_bpe_model = parser.parse_args().w2v_bpe_model
    bpe_model = parser.parse_args().bpe_model
    seq2esm2_dict_file = parser.parse_args().esm2
    
    sp_bpe=[]
    esm2_model=[]
    seq2esm2_dict={}
    if encode_method == 'W2V':
        w2v_model = word2vec.Word2Vec.load(w2v_model)
        os.makedirs(out_path + '/' + deep_method + '/' + encode_method + '_' + str(kmer) + '_' + str(size) + '_' + str(epochs) + '_' + str(window) + '_' + str(sg), exist_ok=True)
        out_path =  out_path + '/' + deep_method + '/' + encode_method + '_' + str(kmer) + '_' + str(size) + '_' + str(epochs) + '_' + str(window) + '_' + str(sg)  
    elif encode_method == 'W2V_BPE':
        kmer=1
        w2v_model = word2vec.Word2Vec.load(w2v_bpe_model)
        sp_bpe = spm.SentencePieceProcessor()
        sp_bpe.Load(bpe_model)
        os.makedirs(out_path + '/' + deep_method + '/' + encode_method, exist_ok=True)
        out_path =  out_path + '/' + deep_method + '/' + encode_method
    elif encode_method == 'BE':
        w2v_model = []
        os.makedirs(out_path + '/' + deep_method + '/' + encode_method, exist_ok=True)
        out_path =  out_path + '/' + deep_method + '/' + encode_method
    elif encode_method == 'NN':
        w2v_model = []
        os.makedirs(out_path + '/' + deep_method + '/' + encode_method, exist_ok=True)
        out_path =  out_path + '/' + deep_method + '/' + encode_method
    elif encode_method == 'ESM2':
        w2v_model = []
        #esm2_model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")
        is_file = os.path.isfile(seq2esm2_dict_file)
        if is_file:
            with open(seq2esm2_dict_file, 'rb') as f:
                seq2esm2_dict = pickle.load(f)
        else:
            print("No seq2esm2_dict.pkl")
            exit()                  
        os.makedirs(out_path + '/' + deep_method + '/' + encode_method, exist_ok=True)
        out_path =  out_path + '/' + deep_method + '/' + encode_method        
        
    else:
        print("no encoding method")  
  
    # cross validation
    for i in range(1, kfold+1):
        train_dataset = trim_input_csv(path + "/" + str(i) + "/cv_train_" + str(i) + ".csv", seqwin, index_col = None)
        val_dataset = trim_input_csv(path + "/" + str(i) + "/cv_val_" + str(i) + ".csv", seqwin, index_col = None)
        test_dataset = trim_input_csv(test_path, seqwin, index_col = None)

        whole_dataset = pd.concat([train_dataset, val_dataset, test_dataset])
        whole_seq = whole_dataset['seq'].tolist()
        #print(len(whole_seq)) #4855
        
        #construction seq to esm2 dictionary
        """
        seq2esm2_dict = {}
        seq2esm2_dict = seq2esm2_dict_construction(whole_seq, esm2_model, aa2index_esm_dict, seqwin)
        # "/home/kurata/myproject/pa3/il6/esm2/seq2esm2_dict.pkl"
        """       
        net = train_test_process(out_path + "/" + str(i), loss_type = loss_type) 
        net.training_testing(train_dataset, val_dataset, test_dataset, deep_method, encode_method, seqwin, kmer, w2v_model, size, sp_bpe, esm2_model, seq2esm2_dict)
        
        output = pd.DataFrame([net.train_probs, net.train_labels], index = ["prob", "label"]).transpose()
        output.to_csv(out_path + "/" + str(i) + "/train_roc.csv")
        output = pd.DataFrame([net.val_probs, net.val_labels], index = ["prob", "label"]).transpose()
        output.to_csv(out_path + "/" + str(i) + "/val_roc.csv")
        
        #independent test
        output = pd.DataFrame([net.test_probs, net.test_labels], index = ["prob", "label"]).transpose()
        output.to_csv(out_path + "/" + str(i) + "/test_roc.csv")

    print('total time:', time.time()-start)


