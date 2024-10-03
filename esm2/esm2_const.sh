#!/bin/sh

in_train=../data/dataset/il6_train.txt
in_test=../data/dataset/il6_test.txt
outfile=il6_seq2esm2_dict.pkl
seqwin=25
n_pop=100

python esm2_const.py --in_train ${in_train} --in_test ${in_test} --outfile ${outfile} --seqwin ${seqwin} --n_pop ${n_pop} 




