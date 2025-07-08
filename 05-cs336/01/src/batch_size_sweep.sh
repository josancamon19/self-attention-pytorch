# sort of stable lr, and finding batch size
# python src/train_transformer.py --batch-size 256 --epochs 1 --lr-max 3e-4 
# python src/train_transformer.py --batch-size 128 --epochs 1 --lr-max 3e-4
# python src/train_transformer.py --batch-size 64 --epochs 1 --lr-max 3e-4

# python src/train_transformer.py --batch-size 640 --epochs 1 --lr-max 3e-4
# python src/train_transformer.py --batch-size 512 --epochs 1 --lr-max 3e-4

# python -m ensurepip --upgrade
# runpod issue: ln -s pip3 .venv/bin/pip


# 64-128 had best results
# 1.44 1.51 < 5% diff
# should consider running both 1 more or 2 more epochs

# python src/train_transformer.py --batch-size 64 --epochs 1 --lr-max 3e-4 -c ".models/gpt2-epoch-1-lr-0.0003-batch-64.pt" --wandb-id "distinctive-voice-27"

# python src/train_transformer.py --batch-size 128 --epochs 1 --lr-max 3e-4 -c ".models/gpt2-epoch-1-lr-0.0003-batch-128.pt" --wandb-id "expert-yogurt-25"

# trying each one of the good batch sizes a bit longer, 2 epochs.
python src/train_transformer.py --batch-size 64 --epochs 2 --lr-max 3e-4
# python src/train_transformer.py --batch-size 96 --epochs 2 --lr-max 3e-4
# python src/train_transformer.py --batch-size 128 --epochs 2 --lr-max 3e-4