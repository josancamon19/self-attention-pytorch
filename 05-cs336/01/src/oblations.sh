python src/train_transformer.py --batch-size 64 --lr-max 4e-3 --adam-weight-decay 0.1 --epochs 1 --lr-warmup-steps 200 --pos-embedding rope --norm-type rms --norm-position pre --ffn-type swiglu

python src/train_transformer.py --batch-size 64 --lr-max 4e-3 --adam-weight-decay 0.1 --epochs 4 --lr-warmup-steps 800 --pos-embedding rope --norm-type rms --norm-position pre --ffn-type swiglu