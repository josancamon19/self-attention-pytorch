import random
import argparse

import dataset
import models
import trainer
import utils

import torch
from tqdm import tqdm

device = "cpu"
if torch.cuda.is_available():
    device = torch.cuda.current_device()
elif torch.backends.mps.is_available() and args.variant == "vanilla":
    device = "mps"

block_size = 128
text = open("wiki.txt", encoding="utf-8").read()
pretrain_dataset = dataset.CharCorruptionDataset(text, block_size)

# We don't suggest you change these hyperparameters, as they're known to work.
# use them for both the vanilla and the RoPE models
mconf = models.GPTConfig(
    pretrain_dataset.vocab_size,
    pretrain_dataset.block_size,
    n_layer=4,
    n_head=8,
    n_embd=256,
)
model = models.GPT(mconf)
model.to(device)
model.load_state_dict(torch.load("vanilla.pretrain.params"))

text = "Khatchig Mouradian "
x = torch.tensor([pretrain_dataset.stoi[s] for s in text], dtype=torch.long)[
    None, ...
].to(device)
pred = utils.sample(model, x, 60, sample=False)[0]
completion = "".join([pretrain_dataset.itos[int(i)] for i in pred])
print("completion", completion)
# TODO, test with ?? tokens, understand this part on evaluate.py, what triggers from pretrained model
# TODO, train again models