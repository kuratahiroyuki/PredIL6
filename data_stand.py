#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import pandas as pd

###########################################
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i1', '--infile1', help='file')
    parser.add_argument('-i2', '--infile2', help='file')
    parser.add_argument('-o1', '--outfile1', help='file')
    parser.add_argument('-o2', '--outfile2', help='file')

    infile1 = parser.parse_args().infile1
    infile2 = parser.parse_args().infile2

    outfile1 = parser.parse_args().outfile1
    outfile2 = parser.parse_args().outfile2

    #sequence preparation
    df1 = pd.read_csv(infile1, sep=',', header=None) #CV
    df2 = pd.read_csv(infile2, sep=',', header=None) #test

    seq11 = df1[0].tolist()
    seq21 = df2[0].tolist()

    #AA = 'ARNDCQEGHILKMFPSTWYVBJOZX'
    seq1=[]
    seq2=[]
    for i in range(len(seq11)):
       if 'B' in seq11[i] or 'J' in seq11[i] or 'O' in seq11[i] or 'Z' in seq11[i] or 'X' in seq11[i]  :
          print('non standard amino acid detected')
       else:
          seq1.append(seq11[i])
          
    for i in range(len(seq21)):
       if 'B' in seq21[i] or 'J' in seq21[i] or 'O' in seq21[i] or 'Z' in seq21[i] or 'X' in seq21[i]  :
          print('non standard amino acid detected')
       else:
          seq2.append(seq21[i])     

    #print(seq1)
    #print(seq2)

    df1[0] = seq1
    df2[0] = seq2

    df1.to_csv(outfile1, header=None, index=None)
    df2.to_csv(outfile2, header=None, index=None)


