import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import transformers as T
import argparse
import tqdm
from utils import *

def apc(x):
    "Perform average product correct, used for contact prediction."
    a1 = x.sum(-1, keepdims=True)
    a2 = x.sum(-2, keepdims=True)
    a12 = x.sum((-1, -2), keepdims=True)

    avg = a1 * a2
    avg.div_(a12)  # in-place to reduce memory
    normalized = x - avg
    return normalized

def attention_extractor(model, data_loader, device,n_sample):
    model.eval()
    all_attentions = np.zeros([n_sample,12,12,512])
    ns=0
    all_cor = []
    with torch.no_grad():
        for e, x in tqdm.tqdm(enumerate(data_loader)):
            x = x.to(device)
            output, attn = model(x,output_attentions = True)
            attn = torch.stack(attn,axis=1)
            attn = attn.to(torch.float16)
            attn = apc(attn) 
            if len(attn.shape) < 4:
                attn = attn.unsqueeze(0)
            
            for i in range(len(attn)):
                ind = torch.where(output[i]>=1)[0].detach().cpu().numpy().tolist() # peak position
                if len(ind)>0:
                    if len(ind)>1:
                        attn_sum = attn[i,:,:,ind,:]
                        attn_sum = attn_sum.sum(axis=2)
                    else:
                        attn_sum = attn[i,:,:,ind,:]
                        
                    attn_sum = attn_sum.squeeze()
                    all_attentions[ns] = attn_sum.detach().cpu().numpy()
                ns+=1
                
        assert ns == n_sample
    all_attentions = all_attentions.astype("float16")
    
    return all_attentions

def main(args):
    device = torch.device("cuda")
    load_path = args.model_path
    fast_tokenizer = T.BertTokenizer.from_pretrained("./model/") 
    model = load_model(fast_tokenizer, load_path)
    print(model)
    model.to(device)
    model.eval()
    
    val_dataset = SequenceDataset(args.file_path, fast_tokenizer)        
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle = False,
        drop_last=False,
        num_workers=2
        )
    
    val_attn = attention_extractor(model, val_loader, device, len(val_dataset))
    np.savez_compressed(args.save_file, attn = val_attn)
    
    # np.savez_compressed(f"{args.save_path}/{args.prefix}_apc_attention.npz",attn = val_attn)
        
parser = argparse.ArgumentParser()
parser.add_argument('--file-path', type=str, help="target")
parser.add_argument('--model-path', type=str, help="save path")
parser.add_argument('--save-file', type=str, help="save path")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
