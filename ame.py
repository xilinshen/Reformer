import pandas as pd
import numpy as np
import h5py
import os
import Bio.Seq as Seq
import subprocess
import tqdm
import sys

### eclip motif
def write_attract_motif(need_id,path=None):
    if path == None:
        path = "./Data/tmp.txt"
        
    motif = open("./Data/pwm_memeformat.txt",'r').readlines()
    write_seq = [i.strip() for i in motif[:9]]
    write = False
    for i in motif:
        i = i.strip()
        if "MOTIF" in i:
            write = False
            id_ = i.split(" ")[1]
            if id_ in need_id:
                write = True
        if write == True:
            write_seq.append(i)
    print(len(write_seq))
    with open(path,'w') as f:
        for i in write_seq:
            print(i,file = f)
            
def load_eclip_data(path):        
    f = h5py.File(path, 'r') # make data

    seq = f['seq'][:]
    strand = f['strand']

    seq = np.array([i.decode() for i in seq])
    strand = np.array([i.decode() for i in strand])

    for e,i in enumerate(strand):
        if i == "-":
            seq[e] = str(Seq.reverse_complement(seq[e]))

    seq = np.stack([[a for a in i] for i in seq])
    return seq, strand

all_prefix_df = pd.read_csv("./Data/ATtRACT_db.csv",index_col=0)

mer=10 ## filter out motifs longer than 10 mer.

def main(prefix, layer, head):    
    if not os.path.exists(f"./Result/{prefix}/"):
        os.mkdir(f"./Result/{prefix}/")

    if not os.path.exists(f"./Result/{prefix}/seq/"):
        os.mkdir(f"./Result/{prefix}/seq/")

    if not os.path.exists(f"./Result/{prefix}/ame/"):
        os.mkdir(f"./Result/{prefix}/ame/")

    seq, strand = load_eclip_data('./Data/{}.h5ad'.format(prefix))
    
    # save attract meme for the gene
    gene = prefix.split("_")[0]
    need_id = list(set(all_prefix_df[all_prefix_df.index == gene].Matrix_id))
    meme_file = f"./Data/{prefix}_attract.txt"
    write_attract_motif(need_id,meme_file)

    # pos seq vs neg seq
    data = np.load("./Data/{}_attention.npz".format(prefix))
    attn = data['attn']
#     latent = data['x']

    attn = attn[:,:,:,2:-1] # remove special token and th first base

    head_attn = attn[:,layer,head,:]

    head_attn_mean = []
    n_sample = []
    n_position = []

    motif_pwm = np.zeros([4,mer])
    for i in range(int(509-mer)):
        head_attn_ = head_attn[:,i:int(i+mer)]

        head_attn_mean.append(head_attn_.mean(axis=1))
        n_sample.extend([i for i in range(len(seq))])
        n_position.extend([i for _ in range(len(seq))])

    head_attn_mean = np.concatenate(head_attn_mean)

    n_sample = np.array(n_sample)
    n_position = np.array(n_position)
    need_ind = np.where(head_attn_mean>np.quantile(head_attn_mean,0.99))[0]
    n_sample = n_sample[need_ind]
    n_position = n_position[need_ind]

    ind = pd.DataFrame({"sample_ind":n_sample,"pos_ind":n_position})
    ind = ind.sort_values(by = ['sample_ind','pos_ind'])

    ind = ind.assign(pos_ind_end = ind.pos_ind+mer)

    ind.to_csv(f"./Result/{prefix}/seq/{prefix}_pos_unmerged.txt",index=None,header=None,sep="\t")
    shell = f"bedtools merge -i ./Result/{prefix}/seq/{prefix}_pos_unmerged.txt > ./Result/{prefix}/seq/{prefix}_pos_merged.txt"
    result = subprocess.getoutput(shell)
    ind_pos = pd.read_table(f"./Result/{prefix}/seq/{prefix}_pos_merged.txt",index_col=None,header = None)

    all_data = pd.DataFrame({"ind":np.arange(len(head_attn)), "start":0,"end":512})
    all_data.to_csv(f"./Result/{prefix}/seq/{prefix}_allseq.txt",index=None,header=None,sep="\t")
    shell = f"bedtools subtract -a ./Result/{prefix}/seq/{prefix}_allseq.txt -b ./Result/{prefix}/seq/{prefix}_pos_unmerged.txt > ./Result/{prefix}/seq/{prefix}_neg_unmerged.txt"
    result = subprocess.getoutput(shell)
    ind_neg = pd.read_table(f"./Result/{prefix}/seq/{prefix}_neg_unmerged.txt",index_col=None,header = None)

    ind_neg = ind_neg[ind_neg.iloc[:,2]-ind_neg.iloc[:,1] >=10]
    ind_pos = ind_pos[ind_pos.iloc[:,2]-ind_pos.iloc[:,1] >=10]

    ind_pos_distribution = np.array(ind_pos.iloc[:,2]-ind_pos.iloc[:,1]) #!
    ind_neg_distribution = np.array(ind_neg.iloc[:,2]-ind_neg.iloc[:,1]) #!

    posfile=f"./Result/{prefix}/seq/layer{layer}_head{head}_pos.fasta"
    with open(posfile,'w')as f:
        for e,(i,start,end) in enumerate(zip(ind_pos.iloc[:,0],ind_pos.iloc[:,1],ind_pos.iloc[:,2])):
            need_seq = "".join(seq[i,start:end])
            print(f">{e}",file=f)
            print(need_seq,file=f)

    negfile=f"./Result/{prefix}/seq/layer{layer}_head{head}_neg.fasta"
    with open(negfile,'w')as f:
        for e in range(len(ind_neg)):
            choice_len = np.random.choice(ind_pos_distribution, 1)[0]
            choice_neg = np.where(ind_neg_distribution>=choice_len)[0]
            if len(choice_neg)==0:
                choice_neg = np.arange(len(ind_neg_distribution))
            neg_ind = np.random.choice(choice_neg,1)[0]
            if ind_neg_distribution[neg_ind]-choice_len == 0:
                choice_start = 0
            else:
                choice_start = np.random.choice(np.arange(ind_neg_distribution[neg_ind]-choice_len),1)[0]
            i,start,end = ind_neg.iloc[neg_ind,0],ind_neg.iloc[neg_ind,1],ind_neg.iloc[neg_ind,2]
            start = start + choice_start
            end = start + choice_len
            need_seq = "".join(seq[i,start:end])
            print(f">{e}",file=f)
            print(need_seq,file=f)

    shell = f"ame --oc ./Result/{prefix}/ame/layer{layer}_head{head}/ --control {negfile} {posfile} {meme_file}"
    r = subprocess.getoutput(shell)
    print(r)
    print(f"ame done: layer{layer} head{head}")

if __name__ == "__main__":
    prefix = sys.argv[1]
    layer = int(sys.argv[2])
    head = int(sys.argv[3])

    if not os.path.exists(f'./Data/{prefix}_attention.npz'):
        shell = f"python attention_extractor.sh {prefix}" 
        r = subprocess.getoutput(shell)
        print(r)
        
    main(prefix,layer,head)
