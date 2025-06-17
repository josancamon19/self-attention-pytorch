"""
Paraphrase detection for GPT starter code.

Consider:
 - ParaphraseGPT: Your implementation of the GPT-2 classification model.
 - train: Training procedure for ParaphraseGPT on the Quora paraphrase detection dataset.
 - test: Test procedure. This function generates the required files for your submission.

Running:
  `python paraphrase_detection.py --use_gpu`
trains and evaluates your ParaphraseGPT model and writes the required submission files.
"""

import argparse
import random
import torch
import os
import numpy as np
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import (
    ParaphraseDetectionDataset,
    ParaphraseDetectionTestDataset,
    load_paraphrase_data,
)
from evaluation import model_eval_paraphrase, model_test_paraphrase
from models.gpt2 import GPT2Model
from transformers import GPT2Model as OpenAIGPT2Model

from optimizer import AdamW
from sonnet_generation import _get_lora_config
from peft import LoraConfig, TaskType, get_peft_model
from types import SimpleNamespace
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import time
import functools
import os
from transformers import GPT2Tokenizer
from torch import autocast

TQDM_DISABLE = False

hf_cache_dir = "./.cache/huggingface"
os.makedirs(hf_cache_dir, exist_ok=True)


def timeit(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.2f} seconds to execute")
        return result
    return wrapper


# Fix the random seed
@timeit
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


@timeit
def cache_model():
    print("Downloading and caching GPT2 tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2", cache_dir=hf_cache_dir)
    # OpenAIGPT2Model.from_pretrained()
    print("Tokenizer cached successfully!")
    

class ParaphraseGPT(nn.Module):
    """Your GPT-2 Model designed for paraphrase detection."""
    
    @timeit
    def __init__(self, args):
        super().__init__()
        self.gpt = GPT2Model.from_pretrained(
            model=args.model_size, d=args.d, l=args.l, num_heads=args.num_heads
        )
        # self.paraphrase_detection_head = nn.Linear(
        #     args.d, 2
        # )  # Paraphrase detection has two outputs: 1 (yes) or 0 (no).

        # By default, fine-tune the full model.
        for param in self.gpt.parameters():
            param.requires_grad = True

        self.generation_config = SimpleNamespace(temperature=0.7, top_p=0.9)

    def forward(self, input_ids, attention_mask, **kwargs):
        gpt_output: dict = self.gpt(input_ids, attention_mask)
        return self.gpt.hidden_state_to_token(gpt_output["last_hidden_state"])[:, -1, :]

        # last_token = gpt_output["last_token"]
        # return self.paraphrase_detection_head(last_token)

    @property
    def config(self):
        return self.gpt.config

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **kwargs):
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def save_model(model, optimizer, args):
    save_info = {
        "model": model.state_dict(),
        "optim": optimizer.state_dict(),
        "args": args,
        "system_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.random.get_rng_state(),
    }

    torch.save(save_info, args.filepath)
    if args.peft:
        model.save_pretrained("./.models/paraphrase")
    print(f"save the model to {args.filepath}")

@timeit
def get_train_datasets(args, is_distributed: bool = False, rank: int = None):
    para_train_data = load_paraphrase_data(args.para_train)
    para_dev_data = load_paraphrase_data(args.para_dev)

    para_train_data = ParaphraseDetectionDataset(para_train_data, args)
    para_dev_data = ParaphraseDetectionDataset(para_dev_data, args)
    
    if is_distributed:
        if rank == 0: # print only once
            world_size = torch.cuda.device_count()
            print(f"Distributed: {len(para_train_data)} total samples")
            print(f"Distributed: ~{len(para_train_data)//world_size} samples per GPU")
            print(f"Distributed: ~{(len(para_train_data)//world_size)//args.batch_size} batches per GPU")
            # print(f"Distributed: Global effective batch size: {args.batch_size * world_size}")
    else:
        print(f"Single GPU: {len(para_train_data)} samples")
        print(f"Single GPU: {len(para_train_data)//args.batch_size} batches total")


    # handle multiple processes
    dev_sampler = DistributedSampler(para_dev_data, shuffle=False) if is_distributed else None
    train_sampler = DistributedSampler(para_train_data, shuffle=True) if is_distributed else None
    # dataloader, grain python.

    para_train_dataloader = DataLoader(
        para_train_data,
        shuffle=(train_sampler is None), # TODO: why?
        batch_size=args.batch_size,
        collate_fn=para_train_data.collate_fn,
        sampler=train_sampler # TODO: what's this for
    )
    para_dev_dataloader = DataLoader(
        para_dev_data,
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=para_dev_data.collate_fn,
        sampler=dev_sampler,
    )
    return para_train_dataloader, para_dev_dataloader



@torch.no_grad()
def test(args):
    """Evaluate your model on the dev and test datasets; save the predictions to disk."""
    device = torch.device("cuda") if args.use_gpu else torch.device("cpu")
    saved = torch.load(args.filepath, weights_only=False)

    model = ParaphraseGPT(saved["args"])
    if args.peft:
        model = get_peft_model(model, _get_lora_config(True))
    else:
        model.load_state_dict(saved["model"])

    model = model.to(device)
    model.eval()
    print(f"Loaded model to test from {args.filepath}")

    para_dev_data = load_paraphrase_data(args.para_dev)
    para_test_data = load_paraphrase_data(args.para_test, split="test")

    para_dev_data = ParaphraseDetectionDataset(para_dev_data, args)
    para_test_data = ParaphraseDetectionTestDataset(para_test_data, args)

    para_dev_dataloader = DataLoader(
        para_dev_data,
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=para_dev_data.collate_fn,
    )
    para_test_dataloader = DataLoader(
        para_test_data,
        shuffle=True,
        batch_size=args.batch_size,
        collate_fn=para_test_data.collate_fn,
    )

    dev_para_acc, _, dev_para_y_pred, _, dev_para_sent_ids = model_eval_paraphrase(
        para_dev_dataloader, model, device
    )
    print(f"dev paraphrase acc :: {dev_para_acc:.3f}")
    test_para_y_pred, test_para_sent_ids = model_test_paraphrase(
        para_test_dataloader, model, device
    )

    with open(args.para_dev_out, "w+") as f:
        f.write(f"id \t Predicted_Is_Paraphrase \n")
        for p, s in zip(dev_para_sent_ids, dev_para_y_pred):
            f.write(f"{p}, {s} \n")

    with open(args.para_test_out, "w+") as f:
        f.write(f"id \t Predicted_Is_Paraphrase \n")
        for p, s in zip(test_para_sent_ids, test_para_y_pred):
            f.write(f"{p}, {s} \n")



if __name__ == "__main__":
    args = get_args()
    args = add_arguments(args)
    os.makedirs("./.models/paraphrase", exist_ok=True)
    args.filepath = f"./.models/paraphrase/{args.model_size}-{args.lr}.pt"
    seed_everything(args.seed)  # Fix the seed for reproducibility.
    args.cache_dir = hf_cache_dir
    
    args.use_bf16 = check_bf16_support() # if not args.distributed else False
    # args.use_bf16 = False
    
    if args.distributed:
        gpus = torch.cuda.device_count()
        print("loading distributed training, gpus:", gpus)
        mp.spawn(train_dist, args=(args, ), nprocs=gpus)
        # use torchrun instead of mp.spawn
        # python process controlling other python processes.
        # fuck, so DDP is a bad idea here, communication overhead is bigger than comp gain.
        # python paraphrase_detection.py --use_gpu --batch_size 160 --distributed, 345 samples second
        # python paraphrase_detection.py --use_gpu --batch_size 52, 100*52/16 = 325 samples per second
        
        # with gradient accumulation 3, 360 samples per second
        # single gpu, gradient accum: 338 samples per second
        
        # maybe here, there's a slight 5/10% gain
        
        # STOPPED TRYING STUFF, let's profile the model
        # nvm profiling is so confusing,
        
        # bf16, 2x as fast, 1/2 memory, wtf. (single GPU)
        # now I'm confused, distributed is not the same? why, wtf
        # now distributed doesn't work at all, after 10% fails, gpu memory full. Prob gradient accumulation, fuck
        
        # TODO: I don't know yet if distributed communication/loss/training is working, solve
    else:
        train(args)
    test(args)
    # 0.86 default settings 10-1e-05-paraphrase.pt
