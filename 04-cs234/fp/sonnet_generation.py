"""
Sonnet generation starter code.

Running:
  `python sonnet_generation.py --use_gpu`

trains your SonnetGPT model and writes the required submission files.
"""

import argparse
import os
import random
import torch

import numpy as np
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2Tokenizer
from einops import rearrange

from datasets import (
    SonnetsDataset,
)
from models.gpt2 import GPT2Model

from optimizer import AdamW
from peft import LoraConfig, TaskType, get_peft_model
from types import SimpleNamespace

TQDM_DISABLE = False


def _get_lora_config(inference: bool, r=32):
    return LoraConfig(
        target_modules=[
            "query",
            "key",
            "value",
            "attention_dense",
            # if only interm_dense, has issue with interm*4, instead of custom *3 of ours, why?
            "gpt.gpt_layers.*.interm_dense",
            "gpt.gpt_layers.*.out_dense",
        ],
        task_type=TaskType.CAUSAL_LM, # semantic label, not functional
        inference_mode=False,
        r=r,
        lora_alpha=r * 2,
        lora_dropout=0.1,
    )


# Fix the random seed.
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class SonnetGPT(nn.Module):
    """Your GPT-2 Model designed for paraphrase detection."""

    def __init__(self, args):
        super().__init__()
        self.gpt = GPT2Model.from_pretrained(
            model=args.model_size, d=args.d, l=args.l, num_heads=args.num_heads
        )
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # By default, fine-tune the full model. TODO: this is maybe not idea.
        for param in self.gpt.parameters():
            param.requires_grad = True

        self.generation_config = SimpleNamespace(temperature=0.7, top_p=0.9)

    @property
    def config(self):
        """Forward the config from the underlying GPT2 model for PEFT compatibility."""
        return self.gpt.config

    def forward(self, input_ids, attention_mask, **kwargs):
        """
        This is similar to the forward for ParaphraseGPT, but we now want to produce a logit for each token in our sequence;
        not just the last token! This will allow our model to learn the natural language distribution that composes sonnets,
        not just the distribution over next tokens for the last token!
        """
        gpt_output = self.gpt(input_ids, attention_mask)
        return self.gpt.hidden_state_to_token(gpt_output["last_hidden_state"])

    def get_device(self):
        for param in self.gpt.parameters():
            return param.device

    @torch.no_grad()
    def generate(self, encoding, temperature=0.7, top_p=0.9, max_length=128):
        """
        Generates an original sonnet using top-p sampling and softmax temperature.

        TODO: this is probably not ideal. You can look at hugging face's model.generate(...) function for inspiration.
        In particular, generating multiple sequences and choosing the best with beam search is one avenue. Top_k is another;
        there are many.
        """
        token_ids = encoding.to(self.get_device())
        attention_mask = torch.ones(token_ids.shape, dtype=torch.int64).to(
            self.get_device()
        )

        for _ in range(max_length):
            # Forward pass to get logits
            logits_sequence = self.forward(token_ids, attention_mask)
            logits_last_token = (
                logits_sequence[:, -1, :] / temperature
            )  # Apply temperature scaling

            # Convert logits to probabilities
            probs = torch.nn.functional.softmax(logits_last_token, dim=-1)

            # Top-p (nucleus) sampling
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            top_p_mask = cumulative_probs <= top_p
            top_p_mask[..., 1:] = top_p_mask[
                ..., :-1
            ].clone()  # Shift mask right for proper thresholding
            top_p_mask[..., 0] = True  # Always include the highest probability token
            filtered_probs = sorted_probs * top_p_mask  # Zero out unlikely tokens
            filtered_probs /= filtered_probs.sum(
                dim=-1, keepdim=True
            )  # Normalize probabilities

            # Sample from filtered distribution
            sampled_index = torch.multinomial(filtered_probs, 1)
            sampled_token = sorted_indices.gather(dim=-1, index=sampled_index)

            # Stop if end-of-sequence token is reached
            if sampled_token.item() == self.tokenizer.eos_token_id:
                break

            # Append sampled token
            token_ids = torch.cat([token_ids, sampled_token], dim=1)
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones((1, 1), dtype=torch.int64).to(self.get_device()),
                ],
                dim=1,
            )

        generated_output = self.tokenizer.decode(token_ids[0].cpu().numpy().tolist())[
            3:
        ]
        return token_ids, generated_output

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **kwargs):
        """Required method for PEFT compatibility with causal LM."""
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }


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
    print(f"save the model to {args.filepath}")
    if args.peft:
        model.save_pretrained("./.models/sonnet")


def train(args):
    """Train GPT-2 for paraphrase detection on the Quora dataset."""
    device = torch.device("cuda") if args.use_gpu else torch.device("cpu")
    # Create the data and its corresponding datasets and dataloader.
    sonnet_dataset = SonnetsDataset(args.sonnet_path)
    sonnet_dataloader = DataLoader(
        sonnet_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        collate_fn=sonnet_dataset.collate_fn,
    )

    # Create the held-out dataset: these only have the first 3 lines. Your job is to fill in the rest!
    held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_path)

    args = add_arguments(args)
    model = SonnetGPT(args)
    model = model.to(device)
    if args.peft:
        model = get_peft_model(model, _get_lora_config(False))
        model.print_trainable_parameters()

    lr = args.lr
    optimizer = AdamW(model.parameters(), lr=lr)

    # Run for the specified number of epochs.
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0

        for batch in tqdm(
            sonnet_dataloader, desc=f"train-{epoch}", disable=TQDM_DISABLE
        ):
            # Get the input and move it to the gpu (I do not recommend training this model on CPU).
            b_ids, b_mask = batch["token_ids"], batch["attention_mask"]
            b_ids = b_ids.to(device)
            b_mask = b_mask.to(device)

            # Compute the loss, gradients, and update the model's parameters.
            optimizer.zero_grad()
            logits = model(b_ids, b_mask)
            logits = rearrange(
                logits[:, :-1].contiguous(), "b t d -> (b t) d"
            )  # Ignore the last prediction in the sequence.
            labels = (
                b_ids[:, 1:].contiguous().flatten()
            )  # Ignore the first token to compose the labels.
            loss = F.cross_entropy(logits, labels, reduction="mean")
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        train_loss = train_loss / num_batches
        print(f"Epoch {epoch}: train loss :: {train_loss:.3f}.")
        # print("Generating several output sonnets...")
        # model.eval()
        # for batch in held_out_sonnet_dataset:
        #     encoding = model.tokenizer(
        #         batch[1], return_tensors="pt", padding=True, truncation=True
        #     ).to(device)
        #     output = model.generate(
        #         encoding["input_ids"], temperature=args.temperature, top_p=args.top_p
        #     )
        #     print(f"{batch[1]}{output[1]}\n\n")

        # TODO: consider a stopping condition to prevent overfitting on the small dataset of sonnets.
        save_model(model, optimizer, args)


@torch.no_grad()
def generate_submission_sonnets(args):
    device = torch.device("cuda") if args.use_gpu else torch.device("cpu")
    saved = torch.load(args.filepath, weights_only=False)

    model = SonnetGPT(saved["args"])
    if args.peft:
        model = get_peft_model(model, _get_lora_config(False))
    # model.load_state_dict(saved["model"])
    model = model.to(device)
    model.eval()

    # Create the held-out dataset: these only have the first 3 lines. Your job is to fill in the rest!
    held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_path)

    generated_sonnets = []
    for batch in held_out_sonnet_dataset:
        sonnet_id = batch[0]
        encoding = model.tokenizer(
            batch[1], return_tensors="pt", padding=False, truncation=True
        ).to(device)
        output = model.generate(
            encoding["input_ids"], temperature=args.temperature, top_p=args.top_p
        )[0][0]
        decoded_output = model.tokenizer.decode(output)
        full_sonnet = f"{decoded_output}\n\n"
        generated_sonnets.append((sonnet_id, full_sonnet))

        print(batch[1], "\noutput:", decoded_output, "-\n----")

    with open(args.sonnet_out, "w+") as f:
        f.write("--Generated Sonnets-- \n\n")
        for sonnet in generated_sonnets:
            f.write(f"\n{sonnet[0]}\n")
            f.write(sonnet[1])


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--sonnet_path", type=str, default="data/sonnets.txt")
    parser.add_argument(
        "--held_out_sonnet_path", type=str, default="data/sonnets_held_out.txt"
    )
    parser.add_argument(
        "--sonnet_out", type=str, default="predictions/generated_sonnets.txt"
    )

    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--peft", action="store_true")

    # Generation parameters.
    parser.add_argument(
        "--temperature", type=float, help="softmax temperature.", default=1.2
    )
    parser.add_argument(
        "--top_p",
        type=float,
        help="Cumulative probability distribution for nucleus sampling.",
        default=0.9,
    )

    parser.add_argument(
        "--batch_size", help="The training batch size.", type=int, default=8
    )
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)
    parser.add_argument(
        "--model_size",
        type=str,
        help="The model size as specified on hugging face.",
        choices=["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"],
        default="gpt2",
    )

    args = parser.parse_args()
    return args


def add_arguments(args):
    """Add arguments that are deterministic on model size."""
    if args.model_size == "gpt2":
        args.d = 768
        args.l = 12
        args.num_heads = 12
    elif args.model_size == "gpt2-medium":
        args.d = 1024
        args.l = 24
        args.num_heads = 16
    elif args.model_size == "gpt2-large":
        args.d = 1280
        args.l = 36
        args.num_heads = 20
    elif args.model_size == "gpt2-xl":
        args.d = 1600
        args.l = 48
        args.num_heads = 25
    else:
        raise Exception(f"{args.model_size} is not supported.")
    return args


if __name__ == "__main__":
    args = get_args()
    os.makedirs("./.models/sonnet", exist_ok=True)
    args.filepath = f"./.models/sonnet/{args.model_size}-{args.lr}.pt"  # Save path.
    seed_everything(args.seed)  # Fix the seed for reproducibility.
    train(args)
    generate_submission_sonnets(args)
    # peft def underperforms here, very little data to train a lot
