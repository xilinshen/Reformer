outdir=./save_reformer_bc/
mkdir ${outdir}

python train_reformer_bc.py \
	--outdir ${outdir} \
	--h5file ./data/example_posneg.h5 \
	--lr 2e-05 \
	--batch-size 64 \
	--epochs 6 \
  --device 0
