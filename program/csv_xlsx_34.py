#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on April 24, 2024
@author: Kurata Laboratory
"""

import openpyxl as px
import pandas as pd
import argparse
import os

columns_measure= ['Threshold', 'Sensitivity', 'Specificity', 'Precision', 'Accuracy', 'MCC', 'F1', 'AUC', 'AUPRC']

if __name__ == '__main__':   
    parser = argparse.ArgumentParser()
    parser.add_argument('--machine_method_1', type=str, help='term')
    parser.add_argument('--encode_method_1', type=str, help='term')
    parser.add_argument('--machine_method_2', type=str, help='term')
    parser.add_argument('--encode_method_2', type=str, help='term')
    parser.add_argument('--outfile', type=str, help='filem')
    parser.add_argument('--species', type=str, help='filem')
    args = parser.parse_args()
    
    """
    machine_method_1 = args.machine_method_1.strip().split(',')
    encode_method_1  = args.encode_method_1.strip().split(',')
    machine_method_2 = args.machine_method_2.strip().split(',')
    encode_method_2  = args.encode_method_2.strip().split(',')
    """
    machine_method_1 = args.machine_method_1.strip().split()
    encode_method_1  = args.encode_method_1.strip().split()
    machine_method_2 = args.machine_method_2.strip().split()
    encode_method_2  = args.encode_method_2.strip().split()
    
    species = args.species
    outfile_name = args.outfile

    infile_name = ["val_measures.csv", "test_measures.csv" ]

    for machine_method in machine_method_1 :
        val_measure=[]
        test_measure=[]
        for i, encode_method in enumerate(encode_method_1):

          infile_path = "../data/result_%s/%s/%s" %(species, machine_method, encode_method)
          infile1 = infile_path + '/' + infile_name[0] #val
          infile2 = infile_path + '/' + infile_name[1] #test

          val_measure.append(  (pd.read_csv(infile1, index_col=0).iloc[-1].values.tolist())) # means
          test_measure.append( (pd.read_csv(infile2, index_col=0).iloc[-1].values.tolist())) # means

        print(f'{val_measure}, {encode_method}')
        
        pd_val_measure  = pd.DataFrame(data=val_measure, index=encode_method_1, columns=columns_measure)
        pd_test_measure = pd.DataFrame(data=test_measure, index=encode_method_1, columns=columns_measure)

        print(pd_val_measure)
        print(pd_test_measure)

        pd_val_test = pd.concat([pd_val_measure, pd_test_measure], axis=0)

        if os.path.exists(outfile_name) == True:
            mode_f ='a'
            with pd.ExcelWriter(outfile_name, engine="openpyxl", mode = mode_f) as writer: 
              pd_val_test.to_excel(writer, sheet_name = machine_method) #index=False, header=False
        else :
            mode_f ='w' 
            with pd.ExcelWriter(outfile_name, engine="openpyxl", mode = mode_f) as writer: 
              pd_val_test.to_excel(writer, sheet_name = machine_method) #index=False, header=False

    for machine_method in machine_method_2 :
        val_measure=[]
        test_measure=[]
        for i, encode_method in enumerate(encode_method_2):

          infile_path = "../data/result_%s/%s/%s" %(species, machine_method, encode_method)
          infile1 = infile_path + '/' + infile_name[0] #val
          infile2 = infile_path + '/' + infile_name[1] #test

          val_measure.append(  (pd.read_csv(infile1, index_col=0).iloc[-1].values.tolist())) # means
          test_measure.append( (pd.read_csv(infile2, index_col=0).iloc[-1].values.tolist())) # means

        pd_val_measure  = pd.DataFrame(data=val_measure, index=encode_method_2, columns=columns_measure)
        pd_test_measure = pd.DataFrame(data=test_measure, index=encode_method_2, columns=columns_measure)

        print(pd_val_measure)
        print(pd_test_measure)

        pd_val_test = pd.concat([pd_val_measure, pd_test_measure], axis=0)

        if os.path.exists(outfile_name) == True:
            mode_f ='a'
            with pd.ExcelWriter(outfile_name, engine="openpyxl", mode = mode_f) as writer: 
              pd_val_test.to_excel(writer, sheet_name = machine_method) #index=False, header=False
        else :
            mode_f ='w' 
            with pd.ExcelWriter(outfile_name, engine="openpyxl", mode = mode_f) as writer: 
              pd_val_test.to_excel(writer, sheet_name = machine_method) #index=False, header=False


    rank_file = './ranking.csv'
    pd_rank = pd.read_csv(rank_file)
    with pd.ExcelWriter(outfile_name, engine="openpyxl", mode = 'a') as writer: 
        pd_rank.to_excel(writer, sheet_name = 'ranking')
        
    """        
    comb_file='../data/result_%s/combine/top_measure.csv' %species
    pd_comb = pd.read_csv(comb_file)
    with pd.ExcelWriter(outfile_name, engine="openpyxl", mode = 'a') as writer: 
        pd_comb.to_excel(writer, sheet_name = 'stack')

    comb_file='../data/result_%s/sel_combine/sel_measure.csv' %species
    pd_comb = pd.read_csv(comb_file)
    with pd.ExcelWriter(outfile_name, engine="openpyxl", mode = 'a') as writer: 
        pd_comb.to_excel(writer, sheet_name = 'select')
 
    maxAUC=0
    maxStack=1
    for i in range(int(pd_comb.shape[0]/2)):
        if pd_comb.loc[2*i,'AUC'] > maxAUC :
            maxAUC=pd_comb.loc[2*i,'AUC']
            maxStack=2*i

    with pd.ExcelWriter(outfile_name, engine="openpyxl", mode = 'a') as writer: 
        pd_comb[maxStack:maxStack+2].to_excel(writer, sheet_name = 'top', index=True) 
    """
        




 


