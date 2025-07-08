# batch-size=640 lr-max=3e-4 ~ 4 epochs.
# for lr in 1e-4 3e-4 5e-4 7e-4 9e-4 1e-3 2e-3 3e-3; do # 3e-3 was best, wtf
# for lr in 4e-3 5e-3 6e-3; do
#     python src/train_transformer.py --batch-size 640 --epochs 1 --lr-max $lr #-ss
# done
# TODO: when unstable, remember to check warmup steps and increase, improved? if not, then def not the right lr

# runpod issue: ln -s pip3 .venv/bin/pip
# kept increasing lr and even 0.005, 0.006 did pretty well, so probably slightly above is the best (by itself), batch size 640

# batch size 64 is best, now test lr alone
for lr in 5e-4 7e-4 9e-4 1e-3 2e-3 3e-3 4e-3; do
    python src/train_transformer.py --batch-size 64 --epochs 1 --lr-max $lr
done