#!/bin/bash

species=md
outsuffix=1
kfold=10
seqwin=35

w2v_path=../w2v_model
#w2v_bpe_model_file=/home/kurata/myproject/common/subword/w2v_bpe/w2v_bpe_400_64_4_50_1.pt
#bpe_model_file=/home/kurata/myproject/common/subword/model/bpe_model_400.model
#esm2_dict_file=/home/kurata/myproject/common/esm2_enc/il6_seq2esm2_dict.pkl 
# Users must build an esm2_dict_file for themselves, because it requires >60GB and it is hard to store here. See the esm2 directory.
# If users remove esm2, they must remove ESM2 from the encoding methods and set "esm2_dict_file=None".

space=" "
machine_method_1="LGBM XGB RF SVM NB KN LR"
encode_method_1="AAC DPC PAAC CTDC CTDT CTDD CKSAAP GAAC GDPC GTPC CTriad BE EAAC AAINDEX BLOSUM62 ZSCALE ESM2" 
w2v_encode="W2V_1 W2V_2"
encode_method_1w=${encode_method_1}$space${w2v_encode}

machine_method_2="TX CNN bLSTM"
encode_method_2="BE NN ESM2"
encode_method_2w=${encode_method_2}$space${w2v_encode}

total_num=148
#total_num = 7*(17+2) + 3*(3+2) = 133+15 = 148  
top_list=20
prefix=combine

cd ..
main_path=`pwd`
echo ${main_path}

########## DATA SETTING ##########

test_fasta=${main_path}/data/dataset/independent_test/independent_test.fa
test_csv=${main_path}/data/dataset/independent_test/independent_test.csv

cd program
cd ml

########## MACHINE LEARNING ##########

train_path=${main_path}/data/dataset/cross_val
result_path=${main_path}/data/result_${species}
esm2_dict=${esm2_dict_file}

for machine_method in ${machine_method_1}
do

    for encode_method in ${encode_method_1}
    do
    kmer=1
    w2v_model=None
    size=-1
    epochs=-1
    window=-1
    sg=-1
    echo ${machine_method} ${encode_method}
    python ml_train_test_46.py  --intrain ${train_path} --intest ${test_csv} --outpath ${result_path} --machine ${machine_method}  --encode ${encode_method} --kfold ${kfold} --seqwin ${seqwin} --kmer ${kmer} --w2vmodel ${w2v_model} --size ${size} --epochs ${epochs} --sg ${sg} --window ${window} --esm2 ${esm2_dict}
    done

    encode_method=W2V
    size=128
    epochs=100
    window=40
    sg=1
    for kmer in 1 2
    do
    w2v_model=${w2v_path}/W2V_${kmer}.pt
    echo ${machine_method} ${encode_method} ${kmer}
    python ml_train_test_46.py  --intrain ${train_path} --intest ${test_csv} --outpath ${result_path} --machine ${machine_method}  --encode ${encode_method} --kfold ${kfold} --seqwin ${seqwin} --kmer ${kmer} --w2vmodel ${w2v_model} --size ${size} --epochs ${epochs} --sg ${sg} --window ${window} --esm2 ${esm2_dict}
    done

done
cd ..
cd network

########## DEEP LEARNING ##########

for deep_method in ${machine_method_2}
do

    for encode_method in ${encode_method_2}
    do
    echo ${deep_method}: ${encode_method}

    if [ $encode_method = BE ]; then
    kmer=1
    size=25
    epochs=-1
    window=-1
    sg=-1
    w2v_model=None
    w2v_bpe_model=None
    bpe_model=None
    esm2_dict=None

    elif [ $encode_method = NN ]; then
    kmer=1
    size=64
    epochs=-1
    window=-1
    sg=-1
    w2v_model=None
    w2v_bpe_model=None
    bpe_model=None
    esm2_dict=None

    elif [ $encode_method = W2V_BPE ]; then 
    kmer=1
    size=64
    epochs=-1
    window=-1
    sg=-1
    w2v_bpe_model=${w2v_bpe_model_file}
    bpe_model=${bpe_model_file}
    w2v_model=None
    esm2_dict=None

    elif [ $encode_method = ESM2 ]; then
    kmer=1
    if [ $deep_method = TX ]; then 
        size=128
    else
        size=1280
    fi
    epochs=-1
    window=-1
    sg=-1
    w2v_model=None
    w2v_bpe_model=None
    bpe_model=None
    esm2_dict=${esm2_dict_file}

    else
    echo no encode method in script
    fi

    python train_test_86.py  --intrain ${train_path} --intest ${test_csv} --outpath ${result_path} --losstype "balanced" --deeplearn ${deep_method}  --encode ${encode_method} --kfold ${kfold} --w2vmodel ${w2v_model} --seqwin ${seqwin} --kmer ${kmer} --size ${size} --epochs ${epochs} --sg ${sg} --window ${window} --w2v_bpe_model ${w2v_bpe_model} --bpe_model ${bpe_model} --esm2 ${esm2_dict}
    done

    encode_method=W2V
    for kmer in 1 2
    do
    echo ${deep_method}: ${encode_method}: ${kmer}
    size=128
    epochs=100
    window=40
    sg=1
    w2v_model=${w2v_path}/W2V_${kmer}.pt
    w2v_bpe_model=None
    bpe_model=None
    esm2_dict=None
    python train_test_86.py  --intrain ${train_path} --intest ${test_csv} --outpath ${result_path} --losstype "balanced" --deeplearn ${deep_method}  --encode ${encode_method} --kfold ${kfold} --w2vmodel ${w2v_model} --seqwin ${seqwin} --kmer ${kmer} --size ${size} --epochs ${epochs} --sg ${sg} --window ${window} --w2v_bpe_model ${w2v_bpe_model} --bpe_model ${bpe_model} --esm2 ${esm2_dict}
    done

done
cd ..

########## ENSEMBLE LEARNING ##########

echo evaluation
python analysis_622.py --machine_method_1 "${machine_method_1}" --encode_method_1 "${encode_method_1w}" --machine_method_2 "${machine_method_2}" --encode_method_2 "${encode_method_2w}" --species ${species} 

outfile=result_${oufsuffix}.xlsx
python csv_xlsx_34.py --machine_method_1 "${machine_method_1}" --encode_method_1 "${encode_method_1w}" --machine_method_2 "${machine_method_2}" --encode_method_2 "${encode_method_2w}" --species ${species} --outfile ${outfile}

echo ensemble
meta=GA
python ml_fusion_GA.py --machine_method_1 "${machine_method_1}" --encode_method_1 "${encode_method_1w}" --machine_method_2 "${machine_method_2}" --encode_method_2 "${encode_method_2w}" --species ${species} --top_list ${top_list}

outfile=result_stack_${outsuffix}.xlsx
python csv_xlsx_37.py --species ${species} --outfile ${outfile} --meta ${meta} --prefix ${prefix}


