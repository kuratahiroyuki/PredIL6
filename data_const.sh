#!/bin/sh

main_path=`pwd`
echo ${main_path}


outfile1=${main_path}/data/dataset/il6_train.txt
outfile2=${main_path}/data/dataset/il6_test.txt

test_fasta=${main_path}/data/dataset/independent_test/independent_test.fa
test_csv=${main_path}/data/dataset/independent_test/independent_test.csv

data_path=${main_path}/data/dataset
kfold=5


#python data_stand.py --infile1 ${infile1} --infile2 ${infile2} --outfile1 ${outfile1} --outfile2 ${outfile2} 

#python data_analysis_1.py --infile1 ${outfile1} --infile2 ${outfile2}

python ${main_path}/train_division_1.py --infile1 ${outfile1} --datapath ${data_path} --kfold ${kfold} 
python ${main_path}/test_fasta.py --infile1 ${outfile2} --outfile1 ${test_fasta} --outfile2 ${test_csv} 



