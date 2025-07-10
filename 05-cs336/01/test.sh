WANDB_MODE=offline python src/train/transformer.py -mp --epochs 1 --dataset owt --use-custom-adam
WANDB_MODE=offline python src/train/transformer.py -mp --epochs 1 --dataset owt --use-custom-adam --use-custom-gradient-clipping
WANDB_MODE=offline python src/train/transformer.py -mp --epochs 1 --dataset owt