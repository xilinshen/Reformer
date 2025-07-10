<div align="center">
<h1>🧬 Reformer</h1>
Driving interpretation of the regulation machanisms underlying RNA processing
</div>

<div align="left">
<h2> 📋 About <a name="about"></a></h2>
Here, we introduce a deep learning pipeline to predict protein-RNA binding affinity purely from sequence data. This pipeline consists of <b>Reformer-BC (RNA-protein binding modeling with transformer - binrary classification)</b> to distinguish binding sites from non-binding sites, followed by <b>Reformer</b>, which predicts binding affinity at single-base resolution. The repository contains the trained model, training script, and the scripts for characterizing protein-RNA binding affinity and prioritizing mutations affecting RNA regulation.

For using the pre-trained model or training models for new RBP targets and cell lines, refer to the installation guide.

<h2> 🔗 Methodology <a name="methodology"></a></h2>
Reformer-BC and Reformer form a deep learning pipline that quantitatively characterizes RNA-protein binding affinity at single-base resolution using cDNA sequences. The journey begins with Reformer-BC, which distinguishes binding sites from non-binding sites, setting the stage for Reformer to refine predictions by incorporating information from relevant regions. 

Developed based on a dataset comprising 225 eCLIP-seq experiments covering 155 RNA binding proteins (RBPs) across 3 cell lines, this approach achieved high prediction accuracy. Reformer excels in identifying protein binding motifs that traditional eCLIP-seq experiments might miss, providing a more nuanced understanding of RNA processing functions.

<div align="center">
<img src="figure/flowchart.png" width="400px">
</div>

<h2> 📖 Installation <a name="Installation"></a></h2>
To use the pre-trained model, clone the hugging face repository:

```bash
sxl/Reformer-BC
sxl/Reformer
```
Required packages can be installed using:

```bash
pip install torch==2.0.1
pip install transformers==4.32.0
pip install h5py==3.9.0
pip install seaborn==0.12.2
pip install kipoiseq==0.7.1
pip install biopython==1.81
pip install pysam==0.21.0
```

<h2> 🌸 De novo training <a name="De novo training"></a></h2>
For de novo training, follow these steps:
</br>
1. Create an output directory;<br>
2. Run the training script:

```bash
## train Reformer-BC
python train_reformer_bc.py \
        --outdir <output_directory> \
        --h5file <training_data> \
        --lr 2e-05 \
        --batch-size 32 \
        --epochs 30 \
        --device 0 1

## train Reformer
python train_reformer.py \
        --outdir <output_directory> \
        --h5file <training_data> \
        --lr 2e-05 \
        --batch-size 32 \
        --epochs 30 \
        --device 0 1
```

Main function arguments:
```bash
--outdir       the output directory
--h5file       input file in h5file format (example: "./data/example.h5") 
--lr           initial learning rate
--batch-size   batch size for training
--epochs       number of training epochs
--device       list of GPU index for training
```
For a complete list of options, run `python train.py -h`.
The full training data can be available at Zendo [doi:10.5281/zenodo.14021440]
<h2> ✨ Predicting protein-RNA binding affinity <a name="Predicting protein-RNA binding affinity"></a></h2> 
To predict protein-RNA binding affinity, follow these steps:

1. Download the pretrained model from xx;

2. Prepare your sequence in h5file format for prediction (example: "./data/test.h5"):
   -   `prefix`: RBP target and cell line name
   -   `code_prefix`: coded target name (example: "./data/prefix_codes.csv")
   -   `seq`: sequence for prediction
   -   `strand`: strand for prediction
</br>
<b> Example usage: </b>

```python
import pandas as pd
import numpy as np
import h5py
import transformers as T
import torch
import tqdm
from utils import *

np.random.seed(42)

# load model
tokenizer = T.BertTokenizer.from_pretrained("./model/") 
model = load_model(tokenizer, "./model/model.bin") # the pretrained model can be download in https://huggingface.co/XLS/Reformer

# load eCLIP coverage
data = h5py.File('./data/test.h5')
strand = data['strand']
dataset = SequenceDataset('./data/test.h5', tokenizer)

# experimental coverage
experiment_coverage = (data['label'][:] * 1e4)/data['coverage'][:].reshape(-1,1) # cpm normalization
experiment_coverage = torch.as_tensor(experiment_coverage[:,1:-2], dtype=torch.float32) # we drop the edge base
experiment_coverage = experiment_coverage.abs()
for i in range(len(experiment_coverage)): 
    if strand[i] == b'-': # reverse complement
        experiment_coverage[i] = experiment_coverage[i].flipud() # flip up-to-down

## prediction
for idx in np.random.choice(np.arange(len(dataset)),10): # we randomly chose 10 data for display
    inputs = dataset.__getitem__(idx)
    output = model(input_ids = inputs.unsqueeze(0).to(model.model.device))
    output = output.detach().cpu().numpy().squeeze()
    plot_tracks({'experimental coverage':experiment_coverage[idx], "prediction":output})
    
```

<h2> ⭐ Motif enrichment with high attention region <a name="Motif enrichment with high attention region"></a></h2>
Motif enrichment is performed on the high attention regions identified by Reformer. This method uncovers significant RNA-binding motifs that may be missed by traditional approaches, providing insights into RNA regulation and protein interactions.</br>
</br>
<b> Example usage: </b>
</br>
<pre><code>
#!/bin/bash
## example: U2AF2 in HepG2
## extract attention scores of peak regions
bash attention_extractor.sh U2AF2_HepG2
## motif enrichment in high attention regions of layer1 head2
mkdir Result
python ame.py U2AF2_HepG2 1 2
</code></pre>
 
<h2> 🔍 Mutation effect prediction <a name="Mutation effect prediction"></a></h2> 
To predict mutation effects on binding affinity, perform the following steps:</br>
</br>1. Specify the RBP and cell line name (example: `./data/prefix_codes.csv` );</br>
</br>2. Specify chromosome, mutation sites, wild-type and mutanted bases, eg. `chrX:133985274:C>T`;</br>
</br>3. We generate wild-type sequence `ref_seq` and mutant sequence `mut_seq` centered on the mutation site.</br>
</br>
<b> Example usage: </b>
</br>

```python
import pandas as pd
import numpy as np
import tqdm
import copy
import pysam
from utils import *

## download reference genome
! wget -P ./data/ https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_24/GRCh38.p5.genome.fa.gz
! gunzip ./data/GRCh38.p5.genome.fa.gz

fasta = pysam.FastaFile("./data/GRCh38.p5.genome.fa")
tokenizer = T.BertTokenizer.from_pretrained("./model/")
model = load_model(tokenizer, "./model/") # the pretrained model can be download in https://huggingface.co/XLS/Reformer

prefix = "PRPF8_HepG2"
snp = "chrX:133985274:C>T"
strand = "-"

ref_seq, mut_seq = snp_2_seq(snp, strand, fasta)

ref_input = tokenize_seq(ref_seq,prefix,tokenizer)
mut_input = tokenize_seq(mut_seq,prefix,tokenizer)

ref_coverage = model(ref_input.to(model.model.device))
mut_coverage = model(mut_input.to(model.model.device))

ref_coverage = ref_coverage.detach().cpu().numpy()
mut_coverage = mut_coverage.detach().cpu().numpy()
```
The predict function returns to `ref_coverage` and `mut_coverage`. The mutation effect is evaluate as the changes in predicted binding affinity between before and after mutation:
```bash
# mutation effect
plot_tracks_comparision({"before mutation":ref_coverage, "after mutation":mut_coverage})

calc_mutation_effect(ref_coverage, mut_coverage)
```

<h2> 🖊️ Citation <a name="citation"></a></h2> 
Shen X, Hou Y, Wang X, Zhang C, Liu J, Shen H, Wang W, Yang Y, Yang M, Li Y, Zhang J, Sun Y, Chen K, Shi L, Li X. A deep learning model for characterizing protein-RNA interactions from sequences at single-base resolution. Patterns (N Y). 2025 Jan 10;6(1):101150. doi: 10.1016/j.patter.2024.101150. 
</div>
