#!/bin/bash

cd ..
cd ..
main_path=`pwd`
echo ${current}
cd program
cd network
species=md
train_path=${main_path}/data/dataset/cross_val
test_csv=${main_path}/data/dataset/independent_test/independent_test.csv
result_path=${main_path}/data/result_${species}
w2v_path=/home/kurata/myproject/common/w2v_model/w2v_g

kfold=5
seqwin=35

machine_method_2="TX CNN bLSTM"
encode_method_2="BE W2V_BPE NN ESM2 W2V_1_128_100_40_1 W2V_2_128_100_40_1 W2V_3_128_100_40_1 W2V_4_128_100_40_1"

for deep_method in TX #TX CNN bLSTM CNN_LSTM
do

    for encode_method in ESM2 #BE NN W2V_BPE ESM2
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
    w2v_bpe_model=/home/kurata/myproject/common/subword/w2v_bpe/w2v_bpe_400_64_4_50_1.pt
    bpe_model=/home/kurata/myproject/common/subword/model/bpe_model_400.model
    w2v_model=None
    esm2_dict=None

    elif [ $encode_method = ESM2 ]; then #ESM-2 model
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
    esm2_dict=/home/kurata/myproject/common/esm2_enc/il13_seq2esm2_dict.pkl

    else
    echo no encode method in script
    fi

    python train_test_86.py  --intrain ${train_path} --intest ${test_csv} --outpath ${result_path} --losstype "balanced" --deeplearn ${deep_method}  --encode ${encode_method} --kfold ${kfold} --w2vmodel ${w2v_model} --seqwin ${seqwin} --kmer ${kmer} --size ${size} --epochs ${epochs} --sg ${sg} --window ${window} --w2v_bpe_model ${w2v_bpe_model} --bpe_model ${bpe_model} --esm2 ${esm2_dict}
    done

encode_method=W2V #general dataset trained W2V mode
for kmer in 1 2 3 4
    do
    echo ${deep_method}: ${encode_method}: ${kmer}
    size=128
    epochs=100
    window=40
    sg=1
    w2v_model=${w2v_path}/W2V_general_${kmer}_128_100_40_1.pt
    w2v_bpe_model=None
    bpe_model=None
    esm2_dict=None
    python train_test_86.py  --intrain ${train_path} --intest ${test_csv} --outpath ${result_path} --losstype "balanced" --deeplearn ${deep_method}  --encode ${encode_method} --kfold ${kfold} --w2vmodel ${w2v_model} --seqwin ${seqwin} --kmer ${kmer} --size ${size} --epochs ${epochs} --sg ${sg} --window ${window} --w2v_bpe_model ${w2v_bpe_model} --bpe_model ${bpe_model} --esm2 ${esm2_dict}
    done

done
