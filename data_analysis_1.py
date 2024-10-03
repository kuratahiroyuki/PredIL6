#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import pickle
import numpy as np
import argparse
import matplotlib.pyplot as plt
 
amino_asids_vector = np.eye(20)
normal_amino_asids = ["A","C","D","E","F","G","H","I","K","L","M","N","P","Q","R","S","T","V","W","Y"]

def file_input_csv(filename,index_col = None):
    data = pd.read_csv(filename, names=['seq','label'], index_col = index_col)
    return data


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--infile1', help='file')
    parser.add_argument('--infile2', help='path')

    in_train = parser.parse_args().infile1
    in_test = parser.parse_args().infile2

    df_train = file_input_csv(in_train, index_col = None)
    df_test  = file_input_csv(in_test, index_col = None)

    df_whole = pd.concat([df_train, df_test])
    whole_seq = df_whole['seq'].tolist()
    seq_len = [ len(i) for i in df_whole['seq'].tolist()]

    print(f'max protein number: {max(seq_len)}') #2527
    #print(f'number of protein seq > 1000: {counter}') #12
    print(f'total protein number: {len(seq_len)}') #90
    print(f'protein length average: {sum(seq_len)/len(seq_len)}') #584

     
	# histgram drawing
    fig = plt.figure(figsize=(10, 5))
    fig.subplots_adjust(bottom=0.15, left=0.15, top=0.9, right=0.9)
    ax = fig.add_subplot(1,1,1)
    
    color_list=['red','green','blue','cyan','magenta','yellow', 'black', 'white', 'orange' ]
    seq_len=np.array(seq_len)
    ax.hist(seq_len, bins=40, linewidth = 1, color= color_list[2]) #width = 10,
    ax.spines["bottom"].set_linewidth(2)
    ax.spines["left"].set_linewidth(2)
    ax.spines["top"].set_linewidth(2)
    ax.spines["right"].set_linewidth(2)

    ax.set_xlim([0,40])
    ax.set_xticks([0,10,20,30,40])
    ax.set_xticklabels([0,10,20,30,40], fontsize=16)
    ax.set_xlabel("Sequence length (-)", fontsize=16)

    ax.set_ylim([0,3000])
    ax.set_yticks([0,1000,2000,3000])
    ax.set_yticklabels([0,1000,2000,3000], fontsize=16)
    ax.set_ylabel("Frequency (-)", fontsize=16)

    plt.savefig("hist_prot", format="pdf", dpi=300)
    #plt.show()


