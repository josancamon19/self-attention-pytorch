# sort of stable lr, and finding batch size
# Run larger batch sizes sequentially

# Run smaller batch sizes in parallel
python src/train_transformer.py --batch-size 256 --epochs 1 --lr-max 3e-4 -ss #&
# python src/train_transformer.py --batch-size 128 --epochs 1 --lr-max 3e-4 -ss &
# python src/train_transformer.py --batch-size 64 --epochs 1 --lr-max 3e-4 -ss &
# wait

python src/train_transformer.py --batch-size 640 --epochs 1 --lr-max 3e-4 -ss
python src/train_transformer.py --batch-size 512 --epochs 1 --lr-max 3e-4 -ss