# sort of stable lr, and finding batch size
# Run larger batch sizes sequentially

# Run smaller batch sizes in parallel
python src/train_transformer.py --batch-size 256 --epochs 1 --lr-max 3e-4 #&
python src/train_transformer.py --batch-size 128 --epochs 1 --lr-max 3e-4
python src/train_transformer.py --batch-size 64 --epochs 1 --lr-max 3e-4
# wait

python src/train_transformer.py --batch-size 640 --epochs 1 --lr-max 3e-4
python src/train_transformer.py --batch-size 512 --epochs 1 --lr-max 3e-4

# python -m ensurepip --upgrade
# runpod issue: ln -s pip3 .venv/bin/pip


# 64-128 had best results
# 1.44 1.51 < 5% diff
# should consider running both 1 more or 2 more epochs