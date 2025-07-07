# initial: python src/train_transformer.py --batch-size 640 --epochs 1 --lr-max 3e-4
for lr in 5e-4 7e-4 1e-3 3e-3; do
    python src/train_transformer.py --batch-size 16 --epochs 1 --lr-max $lr -ss
done
# TODO: test with 10M tokens 3%, should take 1.2 min for each
# TODO: when unstable, remember to check warmup steps and increase, improved? if not, then def not the right lr
