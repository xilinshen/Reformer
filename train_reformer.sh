outdir=./save_reformer/
mkdir ${outdir}

python train_reformer.py \
	--outdir ${outdir} \
	--h5file ./data/example.h5 \
	--lr 2e-05 \
	--batch-size 2 \
	--epochs 30 \
        --device 0
