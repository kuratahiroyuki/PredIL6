#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 19:48:12 2020

@author: kurata
"""

import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
#from pytorch_model_summary import summary
import numpy as np
import copy
import torch
import torch.nn.utils.rnn as rnn

class CNN_LSTM(nn.Module):
    def __init__(self, features, lstm_hidden_size):
        super(CNN_LSTM, self).__init__()
        self.lstm_hidden_size = lstm_hidden_size
        self.conv1 = nn.Conv1d(features, 32, kernel_size=5, stride=1, padding = 2)
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(32, 32, kernel_size=5, stride=1, padding = 2)
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dense = nn.Linear(self.lstm_hidden_size*2, 1)
        self.sigmoid_func = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p = 0.4)
        self.lstm = nn.LSTM(32, self.lstm_hidden_size, batch_first=True, bidirectional=True)

    def forward(self, emb_mat):
        output = torch.transpose(emb_mat, -1, -2)
        output = self.conv1(output)
        output = self.relu(output)
        output = self.maxpool(output)
        output = self.dropout(output)
        
        output = self.conv2(output)
        output = self.relu(output)
        output = self.maxpool(output)
        output = self.dropout(output)
        
        output = torch.transpose(output, -1, -2)
        
        hidden_state, _ = self.lstm(output)
        
        out_final = hidden_state[:, -1][:, :self.lstm_hidden_size]
        out_first = hidden_state[:, 0][:, self.lstm_hidden_size:]
        output = torch.cat([out_final, out_first], dim = 1)
        
        return self.sigmoid_func(self.dense(output))
















