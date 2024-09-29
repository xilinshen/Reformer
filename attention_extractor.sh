prefix=$1

python attention_extractor.py \
--file-path ./data/${prefix}.h5ad \
--model-path ./model/model.bin \
--save-file ./data/${prefix}_attention.npz
