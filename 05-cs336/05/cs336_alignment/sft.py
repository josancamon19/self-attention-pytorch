# finetune based on MATH, improve reasoning ability, rather than correct answers, reasoning trace + answer
# /data/a5-alignment/MATH/sft.jsonl (don't have access)
# in practice, sft is used as a warm start for an RL finetuning step, why?
# - sft requires hq annotated data, RL on ly correct answer,
# - RL can most times even with awesome sft data, find better policies
# ~ not for this model sizes, the 2 processes will be treated separately

import pdb
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
import torch

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-Math-1.5B",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load DART-Math-Uniform dataset
dataset = load_dataset("hkust-nlp/dart-math-uniform")
train_dataset = dataset["train"]


class MathDataset(Dataset):
    def __init__(
        self,
        dataset,
        tokenizer,
        max_length=1024,
        prompt_template_path: str = "cs336_alignment/prompts/r1_zero.prompt",
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        with open(prompt_template_path, "r") as f:
            self.prompt_template = f.read().strip()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        query = item["query"]
        response = item["response"]
        thinking, answer = [p.strip() for p in response.split("The answer is:")]
        answer = f"The answer is: {answer}"

        prompt = f"{self.prompt_template.replace('{question}', query)}{thinking}</think><answer>{answer}</answer>"
        print(prompt)
        # Tokenize
        encoding = self.tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()

        return {
            "input_ids": input_ids[:-1],
            "attention_mask": attention_mask[:-1],
            "labels": input_ids.clone()[1:],
        }


# Create dataset and dataloader
math_dataset = MathDataset(train_dataset, tokenizer)
data_loader = DataLoader(math_dataset, batch_size=1, shuffle=True)


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    pass  # cross entropy? why not?
    # Logging per-token entropies. When doing RL, it is often useful to keep track of per-token entropies to
    # see if the predictive distribution of the model is becoming (over)confident. We will implement this now and
    # compare how each of our finetuning approaches affects the model’s predictive entropy.
    # The entropy of a discrete distribution p(x) with support Xis defined as
    # H(p) =−
    # x∈X
    # p(x) log p(x).


def finetune():
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.train()
    model.to(device)
    accum_steps = 4

    for idx, batch in enumerate(data_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        pdb.set_trace()
        break
        loss = outputs.loss
        loss.backward()

        if (idx + 1) % accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        if idx % 100 == 0:
            print(f"Step {idx}, Loss: {loss.item():.4f}")
