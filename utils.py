import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import AdamW
import torch.utils.data
import transformers as T

import Bio.Seq
import numpy as np
import h5py
import pysam 
import copy
import pandas as pd

def load_model(tokenizer, model_path):
    model = Bert4Coverage(tokenizer)
    
    state_dict = torch.load(model_path, map_location  = "cpu")
    model.load_state_dict(state_dict, strict = False)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    return model
    
def snp_2_seq(snp, strand, fasta):
    chromosome =  snp.split(":")[0]
    position = int(snp.split(":")[1])
    ref = snp.split(":")[2].split(">")[0]
    mut = snp.split(":")[2].split(">")[1]
    
    start = position - 256
    end = position + 256
    
    seq = fasta.fetch(chromosome, start, end)
    ref_seq = copy.copy(seq)
    assert ref_seq[255] == ref
    
    mut_seq = [i for i in seq]
    mut_seq[255] = mut
    mut_seq = "".join(mut_seq)
    
    if strand == "-":
        ref_seq = str(Bio.Seq.Seq(ref_seq).reverse_complement())
        mut_seq = str(Bio.Seq.Seq(mut_seq).reverse_complement())
    return ref_seq, mut_seq

def tokenize_seq(ss,prefix,tokenizer):
    prefix_code = pd.read_csv('./data/prefix_codes.csv')
    prefix_code_dic = {a:b for a,b in zip(prefix_code.prefix, prefix_code.code_prefix)}
    
    ss = [ss[i:int(i+3)] for i in range(int(len(ss)-2))]# 3 mer 
    seq = [prefix_code_dic[prefix]]
    seq.extend(ss[:-1])
    inputs = tokenizer(seq, is_split_into_words=True, add_special_tokens=True, return_tensors='pt') 
    return inputs['input_ids']

def calc_mutation_effect(wt_coverage, mt_coverage):
    # the mutation effect was evaluate as the changes in predicted binding affinity before and after mutation 
    if len(mt_coverage.shape) == 1:
        mt_coverage = mt_coverage.reshape(1,-1)
    if len(wt_coverage.shape) == 1:
        wt_coverage = wt_coverage.reshape(1,-1)
        
    peak_mutpos_diff = np.abs(wt_coverage[:,202:302].sum(axis=1) - mt_coverage[:,202:302].sum(axis=1))/wt_coverage[:,202:302].sum(axis=1) # the binding affinity was measured as the coverage summation of 100 bp around the mutated nucleotide
    return peak_mutpos_diff

def plot_tracks(tracks, interval=None, height=1.5):
    # tracks : {"track1":np.array([...])} 
    # interval : "chr1:1-10000"

    from matplotlib import pyplot as plt
    import seaborn as sns
    import kipoiseq
    import numpy as np
    
    if interval == None:
        plot_interval = False
        n = [i for i in tracks.values()][0]
        interval= kipoiseq.Interval('xx', 0,len(n))
    else:
        plot_interval = True
        start=interval.split(":")[1].split("-")[0]
        end=interval.split(":")[1].split("-")[1]
        chr_=interval.split(":")[0]
        interval = kipoiseq.Interval(chr_, start,end)

    fig, axes = plt.subplots(len(tracks), 1, figsize=(20, height * len(tracks)), sharex=True)
    if len(tracks)>=2:
        for ax, (title, y) in zip(axes, tracks.items()):
            ax.fill_between(np.linspace(interval.start, interval.end, num=len(y)), y)
            ax.set_title(title)
            sns.despine(top=True, right=True, bottom=True)
            if plot_interval == True: 
                ax.set_xlabel(str(interval))
            plt.tight_layout()
    else:
        ax = axes
        for (title, y) in tracks.items():
            ax.fill_between(np.linspace(interval.start, interval.end, num=len(y)), y)
            ax.set_title(title)
            sns.despine(top=True, right=True, bottom=True)
            if plot_interval == True:
                ax.set_xlabel(str(interval))
            plt.tight_layout()

def plot_tracks_comparision(tracks, interval=None, height=1.5):
    # tracks : {"track1":np.array([...])} 
    # interval : "chr1:1-10000"

    from matplotlib import pyplot as plt
    import seaborn as sns
    import kipoiseq
    import numpy as np

    if interval == None:
        plot_interval = False
        n = [i for i in tracks.values()][0]
        interval= kipoiseq.Interval('xx', 0,len(n))
    else:
        plot_interval = True
        start=interval.split(":")[1].split("-")[0]
        end=interval.split(":")[1].split("-")[1]
        chr_=interval.split(":")[0]
        interval = kipoiseq.Interval(chr_, start,end)

    fig, axes = plt.subplots(1, 1, figsize=(20, height * len(tracks)), sharex=True)
    ax = axes
    for (title, y) in tracks.items():
        ax.fill_between(np.linspace(interval.start, interval.end, num=len(y)), y,alpha=0.5, label = title)
        # ax.set_title(title)
        sns.despine(top=True, right=True, bottom=True)
        if plot_interval == True: 
            ax.set_xlabel(str(interval))
        plt.tight_layout()
    plt.legend()

class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, h5file, tokenizer, max_length=512, train=False):
        df = h5py.File(h5file)
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.sequence = df['seq']
        self.barcode = df['code_prefix']
        self.strand = np.array(df['strand'])

        self.n = len(self.sequence)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        ss = self.sequence[i].decode()
        if self.strand[i] == b'-':
            ss = Bio.Seq.reverse_complement(ss)
        
        ss = [ss[i:int(i+3)] for i in range(int(len(ss)-2))]# 3 mer data
        seq = [self.barcode[i].decode()]
        seq.extend(ss[:-1])
        inputs = self.tokenizer(seq, is_split_into_words=True, add_special_tokens=True, return_tensors='pt') 
        return inputs['input_ids']
        
class SequenceDataset4train(torch.utils.data.Dataset):
    def __init__(self, h5file, tokenizer, max_length=512, train=False):
        df = h5py.File(h5file)
        self.tokenizer = tokenizer
        self.max_length = max_length

        if train:
            self.sequence = df['trn_seq']
            self.label = df['trn_label']
            self.barcode = df['trn_code_prefix']
            self.coverage = np.array(df['trn_coverage'])
            self.strand = np.array(df['trn_strand'])
        else:
            self.sequence = df['val_seq']
            self.label = df['val_label']
            self.barcode = df['val_code_prefix']
            self.coverage = np.array(df['val_coverage'])
            self.strand = np.array(df['val_strand'])

        self.n = len(self.label)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        experiment_coverage = torch.tensor(self.label[i] * 1e4 / self.coverage[i]) # do cpm normalization
        ss = self.sequence[i].decode()
        if self.strand[i] == b'-':
            ss = Bio.Seq.reverse_complement(ss)
            experiment_coverage = experiment_coverage.flipud() # flip up-to-down
        experiment_coverage.abs_()           # scores on minus strand is negative
        
        ss = [ss[i:int(i+3)] for i in range(int(len(ss)-2))]# 3 mer data
        seq = [self.barcode[i].decode()]
        seq.extend(ss[:-1])
        inputs = self.tokenizer(seq, is_split_into_words=True, add_special_tokens=True, return_tensors='pt') 
        experiment_coverage = torch.tensor(experiment_coverage)[1:-2] # 3 mer label
        return inputs['input_ids'], torch.as_tensor(experiment_coverage, dtype=torch.float32)

class Bert4Coverage(nn.Module):
    def __init__(self,tokenizer):
        super(Bert4Coverage, self).__init__()
        config = T.BertConfig('./model/config.json')
        config.vocab_size = np.max([len(tokenizer),512])
        
        self.model = T.BertModel(config)
        self.model.resize_token_embeddings(len(tokenizer))
        hidden_size = self.model.config.hidden_size #768
        self.dropout = nn.Dropout(0.2)
        self.lin = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, input_ids):
        hidden = self.model(input_ids=input_ids.squeeze(1)).last_hidden_state[:,2:-1,:]
        hidden = self.dropout(hidden)
        score = self.lin(hidden).squeeze()
        return score.relu()