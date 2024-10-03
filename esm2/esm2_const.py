
import torch
#import esm
import pandas as pd
import time
import argparse
import pickle
#ESM-2
aa2index_esm2 = {'A': 5, 'R': 10, 'N': 17, 'D': 13, 'C': 23, 'E': 9, 'Q': 16, 'G': 6, 'H': 21, 'I': 12, 'L': 4, 'K': 15, 'M': 20, 'F': 18, 'P': 14, 'S': 8, 'T': 11, 'W': 22, 'Y': 19, 'V': 7, 'X': 1, 'B': 25, 'O': 28, 'U': 26, 'Z': 27, '-': 1}


def trim_input_csv(filename, seqwin, index_col = None):
    df1 = pd.read_csv(filename, delimiter=',', names=['seq','label'],index_col = index_col)
    seq = df1.loc[:,'seq'].tolist()
    #data triming and padding
    for i in range(len(seq)):
       if len(seq) > seqwin:
         seq[i]=seq[i][0:seqwin]
    for i in range(len(seq)):
       df1.loc[i,'seq'] = seq[i]
         
    return df1


def seq2esm2_dict_construct(whole_seq, esm2_model, aa2index_esm2 , seqwin):    
    esm2_model.eval()  # disables dropout for deterministic results    
    tokens = []
    for seq in whole_seq:
        tok =[0]
        seq_len = len(seq)
        for aa in seq:
            tok += [aa2index_esm2 [aa]]
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
    
    seq2esm_dict = {}  
    for i in range(token_representations.shape[0]):
        seq2esm_dict[whole_seq[i]] = token_representations[i,:,:].squeeze(0)
        
    return seq2esm_dict

def div_construct(seq2esm2_dict, whole_seq, esm2_model, aa2index_esm2 , seqwin, n_pop):

    n_sample = len(whole_seq)
    print(f'n_sample: {n_sample}')#341, 3104
    
    n_group = int(n_sample/n_pop)
    n_extra = int(n_sample%n_pop)
    for i in range(n_group):
        print(i)
        group_seq = whole_seq[i*n_pop:(i+1)*n_pop]   
        x = seq2esm2_dict_construct(group_seq, esm2_model, aa2index_esm2, seqwin)
        seq2esm2_dict.update(x)
    if n_extra > 0:
        group_seq = whole_seq[n_group*n_pop:]
        x = seq2esm2_dict_construct(group_seq, esm2_model, aa2index_esm2, seqwin)
        seq2esm2_dict.update(x)
    
    return seq2esm2_dict
      
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_train', help='file')
    parser.add_argument('--in_test', help='path')
    parser.add_argument('--outfile', help='path')
    parser.add_argument('--seqwin', type=int, help='value')
    parser.add_argument('--n_pop', type=int, help='value')

    in_train = parser.parse_args().in_train
    in_test = parser.parse_args().in_test
    outfile = parser.parse_args().outfile
    seqwin = parser.parse_args().seqwin
    n_pop = parser.parse_args().n_pop

    esm2_model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")
    #batch_converter = alphabet.get_batch_converter()
   
    df_train = trim_input_csv(in_train, seqwin, index_col = None)
    df_test  = trim_input_csv(in_test, seqwin, index_col = None)
    print(df_test)
    df_whole = pd.concat([df_train, df_test])
    whole_seq = df_whole['seq'].tolist()
    
    start_time=time.time()
    
    seq2esm2_dict = {}        
    seq2esm2_dict = div_construct(seq2esm2_dict, whole_seq, esm2_model, aa2index_esm2, seqwin, n_pop)

    print(len(seq2esm2_dict))
    with open(outfile, 'wb') as f:
        pickle.dump(seq2esm2_dict, f)
    
    print(time.time()-start_time)
    
        
    

