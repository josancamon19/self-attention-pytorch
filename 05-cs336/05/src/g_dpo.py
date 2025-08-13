import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple
from src.b_sft import get_response_log_probs
import torch
from torch.utils.data import Dataset, DataLoader
import random
import argparse
from dataclasses import dataclass
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import wandb


def _parse_first_turn_from_string(convo_text: str) -> Tuple[str, str] | None:
    """
    Extract (instruction, assistant_reply) from a transcript string that uses
    'Human:' and 'Assistant:' markers. Returns None if multi-turn or malformed.
    """
    if not isinstance(convo_text, str):
        return None

    # Must be exactly one human turn
    human_occurrences = re.findall(r"\bHuman:\s*", convo_text)
    if len(human_occurrences) != 1:
        return None

    # Capture the first human message and the first assistant reply that follows it
    m = re.search(
        r"Human:\s*(.*?)\s*Assistant:\s*(.*?)(?=\n\s*Human:|\Z)",
        convo_text,
        flags=re.DOTALL,
    )
    if not m:
        return None

    instruction = m.group(1).strip()
    assistant = m.group(2).strip()
    if not instruction or not assistant:
        return None

    return instruction, assistant


def _parse_first_turn_from_messages(messages: Any) -> Tuple[str, str] | None:
    """
    Extract (instruction, assistant_reply) from a list of message dicts like:
      [{'role': 'human'|'user'|'assistant', 'content': '...'}, ...]
    Returns None if multi-turn (more than one human/user) or malformed.
    """
    if not isinstance(messages, list):
        return None

    # Count human turns
    def is_human(role: str) -> bool:
        return role.lower() in {"human", "user"}

    def is_assistant(role: str) -> bool:
        return role.lower() in {"assistant"}

    human_indices = [
        i for i, m in enumerate(messages) if is_human(str(m.get("role", "")))
    ]
    if len(human_indices) != 1:
        return None

    h_idx = human_indices[0]
    instruction = str(messages[h_idx].get("content", "")).strip()
    if not instruction:
        return None

    # Find the first assistant message after the first human
    assistant_msg = None
    for m in messages[h_idx + 1 :]:
        if is_assistant(str(m.get("role", ""))):
            assistant_msg = str(m.get("content", "")).strip()
            break

    if not assistant_msg:
        return None

    return instruction, assistant_msg


def _extract_first_turn(convo: Any) -> Tuple[str, str] | None:
    """
    Supports both string conversations ('Human:/Assistant:') and message arrays.
    Returns (instruction, assistant_reply) or None if invalid/multi-turn.
    """
    # Common patterns seen in public HH datasets:
    # - convo as a string with 'Human:'/'Assistant:' markers
    # - convo as dict {'messages': [...]}
    # - convo as list of {'role','content'}
    if isinstance(convo, str):
        return _parse_first_turn_from_string(convo)

    if isinstance(convo, dict):
        if "messages" in convo:
            return _parse_first_turn_from_messages(convo["messages"])
        # Some variants: {'human': '...', 'assistant': '...'} but those are not the HH RLHF format.
        # Fall through to None.

    if isinstance(convo, list):
        return _parse_first_turn_from_messages(convo)

    return None


def load_anthropic_hh_dpo_dataset(
    data_dir: str | os.PathLike | None = None,
    files: List[str] | None = None,
) -> List[Dict[str, str]]:
    """
    Load and combine the Anthropic HH RLHF training data from four JSONL files, keeping only single-turn examples.

    Processing:
    - Ignore multi-turn conversations (more than one human message).
    - For each example, extract:
        instruction: first human message
        chosen: assistant reply from the 'chosen' conversation
        rejected: assistant reply from the 'rejected' conversation
    - Record 'source' as the originating filename.

    Returns:
        List of dicts with keys: 'instruction', 'chosen', 'rejected', 'source'
    """
    repo_root = Path(__file__).resolve().parent.parent
    if data_dir is None:
        data_dir = repo_root / "data" / "rlhf"
    data_dir = Path(data_dir)

    if files is None:
        files = [
            "helpful-rejection-sampled.train.jsonl",
            "helpful-online.train.jsonl",
            "helpful-base.train.jsonl",
            "harmless-base.train.jsonl",
        ]

    combined: List[Dict[str, str]] = []

    for fname in files:
        fpath = data_dir / fname
        if not fpath.is_file():
            continue

        with fpath.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    obj = json.loads(line)
                except Exception:
                    continue

                chosen = obj.get("chosen")
                rejected = obj.get("rejected")

                ch = _extract_first_turn(chosen)
                rj = _extract_first_turn(rejected)
                if not ch or not rj:
                    continue

                instruction_ch, chosen_reply = ch
                instruction_rj, rejected_reply = rj

                # Ensure both conversations share the same first prompt
                if (
                    instruction_ch
                    and instruction_rj
                    and instruction_ch != instruction_rj
                ):
                    continue

                instruction = instruction_ch or instruction_rj
                if not instruction or not chosen_reply or not rejected_reply:
                    continue

                combined.append(
                    {
                        "instruction": instruction,
                        "chosen": chosen_reply,
                        "rejected": rejected_reply,
                        "source": fpath.name,
                    }
                )

    return combined


def per_instance_dpo_loss(
    model,
    model_ref,
    tokenizer,
    beta,
    prompt,
    resp_chose,
    resp_rejected,
):
    # Build templated sequences with EOS per SFT convention
    prompt_path = Path(__file__).resolve().parent / "prompts" / "alpaca_sft.prompt"
    with open(prompt_path, "r") as f:
        template = f.read().strip()

    def build_ids(text_prompt: str, response: str) -> torch.Tensor:
        text = template.format(instruction=text_prompt, response=response)
        ids = tokenizer.encode(text, add_special_tokens=True)
        eos_id = tokenizer.eos_token_id
        if eos_id is not None and (len(ids) == 0 or ids[-1] != eos_id):
            ids = ids + [eos_id]
        return torch.tensor([ids], dtype=torch.long)

    chosen_ids = build_ids(prompt, resp_chose)
    rejected_ids = build_ids(prompt, resp_rejected)

    # Helper: unconditional log-prob under a model for a single sequence
    def seq_logprob(
        cur_model, full_ids: torch.Tensor, requires_grad: bool
    ) -> torch.Tensor:
        device = next(cur_model.parameters()).device
        input_ids = full_ids[:, :-1].to(device)
        labels = full_ids[:, 1:].to(device)
        if requires_grad:
            out = get_response_log_probs(
                cur_model, input_ids, labels, return_token_entropy=False
            )["log_probs"]
        else:
            with torch.no_grad():
                out = get_response_log_probs(
                    cur_model, input_ids, labels, return_token_entropy=False
                )["log_probs"]
        # sum over time, keep as scalar on model device
        return out.sum()

    # Compute log-probs for both models
    l_m_chosen = seq_logprob(model, chosen_ids, requires_grad=True)
    l_m_rejected = seq_logprob(model, rejected_ids, requires_grad=True)
    l_ref_chosen = seq_logprob(model_ref, chosen_ids, requires_grad=False)
    l_ref_rejected = seq_logprob(model_ref, rejected_ids, requires_grad=False)

    delta_model = l_m_chosen - l_m_rejected
    delta_ref = l_ref_chosen - l_ref_rejected
    device = next(model.parameters()).device
    beta_t = torch.as_tensor(beta, device=device, dtype=delta_model.dtype)

    # DPO loss: -log(sigmoid(beta * [(Δ_model) - (Δ_ref)]))
    logits = beta_t * (delta_model.to(device) - delta_ref.to(device))
    loss = -torch.nn.functional.logsigmoid(logits)
    return loss


if __name__ == "__main__":
    examples = load_anthropic_hh_dpo_dataset()

# ==============================
# ======= DPO Training =========
# ==============================


@dataclass
class PreferenceExample:
    instruction: str
    chosen: str
    rejected: str


class PreferenceDataset(Dataset):
    def __init__(self, rows: List[Dict[str, str]]):
        self.rows = [
            PreferenceExample(
                instruction=r["instruction"], chosen=r["chosen"], rejected=r["rejected"]
            )
            for r in rows
            if "instruction" in r and "chosen" in r and "rejected" in r
        ]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> PreferenceExample:
        return self.rows[idx]


def split_train_val(
    rows: List[Dict[str, str]], val_fraction: float = 0.05, seed: int = 42
):
    rng = random.Random(seed)
    shuffled = rows[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_fraction))
    val_rows = shuffled[:n_val]
    train_rows = shuffled[n_val:]
    return train_rows, val_rows


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_pad_token(model, tokenizer):
    # Align pad token between tokenizer and model config when missing
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model_config = getattr(model, "config", None)
    if model_config is not None and getattr(model_config, "pad_token_id", None) is None:
        if tokenizer.pad_token_id is not None:
            model_config.pad_token_id = tokenizer.pad_token_id


@torch.no_grad()
def compute_preference_accuracy(
    model,
    tokenizer,
    dataset: PreferenceDataset,
    max_eval_samples: int | None = 512,
) -> float:
    model_device = next(model.parameters()).device
    n = len(dataset)
    total = 0
    correct = 0
    eval_indices = list(range(n))
    if max_eval_samples is not None:
        eval_indices = eval_indices[:max_eval_samples]

    prompt_path = Path(__file__).resolve().parent / "prompts" / "alpaca_sft.prompt"
    with open(prompt_path, "r") as f:
        template = f.read().strip()

    def build_ids(text_prompt: str, response: str) -> torch.Tensor:
        text = template.format(instruction=text_prompt, response=response)
        ids = tokenizer.encode(text, add_special_tokens=True)
        eos_id = tokenizer.eos_token_id
        if eos_id is not None and (len(ids) == 0 or ids[-1] != eos_id):
            ids = ids + [eos_id]
        return torch.tensor([ids], dtype=torch.long)

    for i in eval_indices:
        ex = dataset[i]
        # Policy-only pairwise accuracy: log p(y_w) > log p(y_l)
        chosen_ids = build_ids(ex.instruction, ex.chosen)
        rejected_ids = build_ids(ex.instruction, ex.rejected)

        input_ids = chosen_ids[:, :-1].to(model_device)
        labels = chosen_ids[:, 1:].to(model_device)
        l_m_chosen = get_response_log_probs(model, input_ids, labels)["log_probs"].sum()

        input_ids = rejected_ids[:, :-1].to(model_device)
        labels = rejected_ids[:, 1:].to(model_device)
        l_m_rejected = get_response_log_probs(model, input_ids, labels)[
            "log_probs"
        ].sum()

        correct += int((l_m_chosen - l_m_rejected).item() > 0)
        total += 1

    return float(correct) / max(1, total)


def dpo_train_epoch(
    model,
    model_ref,
    tokenizer,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    beta: float,
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float | None = 1.0,
    val_ds=None,
    validation_every_steps: int = 100,
    save_every_steps: int = 500,
    max_eval_samples: int | None = 512,
    output_dir: str = None,
    global_step: int = 0,
) -> dict:
    model.train()
    model_ref.eval()

    running_loss = 0.0
    steps = 0
    optimizer.zero_grad(set_to_none=True)

    num_batches = len(train_loader)
    bs_guess = getattr(train_loader, "batch_size", None) or 1
    total_micro = num_batches * bs_guess

    best_val_acc = -1.0
    current_step = global_step

    with tqdm(total=total_micro, desc="train", leave=False) as pbar:
        for b_idx, batch in enumerate(train_loader):
            # batch is a list[PreferenceExample]
            micro_loss = 0.0
            for m_idx, ex in enumerate(batch, start=1):
                loss = per_instance_dpo_loss(
                    model=model,
                    model_ref=model_ref,
                    tokenizer=tokenizer,
                    beta=beta,
                    prompt=ex.instruction,
                    resp_chose=ex.chosen,
                    resp_rejected=ex.rejected,
                )
                (loss / gradient_accumulation_steps).backward()
                micro_loss += loss.item()

                pbar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "batch": f"{b_idx + 1}/{num_batches}",
                        "micro": f"{m_idx}/{len(batch)}",
                        "step": current_step,
                    },
                    refresh=False,
                )
                pbar.update(1)

                # Log loss per example to wandb
                wandb.log({"train_loss_per_example": loss.item()})

            if max_grad_norm is not None and max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            steps += 1
            current_step += 1
            batch_avg_loss = micro_loss / max(1, len(batch))
            running_loss += batch_avg_loss

            # Log batch-level metrics to wandb
            wandb.log(
                {"train_loss_per_batch": batch_avg_loss, "batch_step": current_step}
            )

            # Validation check
            if current_step % validation_every_steps == 0:
                val_acc = compute_preference_accuracy(
                    model=model,
                    tokenizer=tokenizer,
                    dataset=val_ds,
                    max_eval_samples=max_eval_samples,
                )
                print(f"\nStep {current_step} - val_acc={val_acc:.4f}")
                wandb.log({"val_accuracy": val_acc})

                # Save if best
                if val_acc > best_val_acc and output_dir is not None:
                    best_val_acc = val_acc
                    os.makedirs(output_dir, exist_ok=True)
                    best_path = os.path.join(output_dir, f"best_step_{current_step}")
                    if hasattr(model, "save_pretrained"):
                        model.save_pretrained(best_path)
                    if hasattr(tokenizer, "save_pretrained"):
                        tokenizer.save_pretrained(best_path)
                    print(f"Saved best model to {best_path}")

            # Regular save check
            elif output_dir is not None and current_step % save_every_steps == 0:
                os.makedirs(output_dir, exist_ok=True)
                save_path = os.path.join(output_dir, f"checkpoint_step_{current_step}")
                if hasattr(model, "save_pretrained"):
                    model.save_pretrained(save_path)
                if hasattr(tokenizer, "save_pretrained"):
                    tokenizer.save_pretrained(save_path)
                print(f"\nSaved checkpoint to {save_path}")

    avg_loss = running_loss / max(1, steps)
    return {
        "train_loss": avg_loss,
        "train_steps": steps,
        "global_step": current_step,
        "best_val_acc": best_val_acc,
    }


def collate_list(batch: List[PreferenceExample]) -> List[PreferenceExample]:
    return batch


def train_dpo(
    model,
    model_ref,
    tokenizer,
    train_rows: List[Dict[str, str]],
    val_rows: List[Dict[str, str]],
    output_dir: str,
    beta: float = 0.1,
    lr: float = 5e-6,
    batch_size: int = 4,
    epochs: int = 1,
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float | None = 1.0,
    max_eval_samples: int | None = 512,
    validation_every_steps: int = 100,
    save_every_steps: int = 500,
    seed: int = 42,
) -> dict:
    set_seed(seed)
    ensure_pad_token(model, tokenizer)
    ensure_pad_token(model_ref, tokenizer)

    # Initialize wandb
    wandb.init(
        project="assignment-05-dpo",
        config={
            "beta": beta,
            "lr": lr,
            "batch_size": batch_size,
            "epochs": epochs,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "max_grad_norm": max_grad_norm,
            "validation_every_steps": validation_every_steps,
            "save_every_steps": save_every_steps,
            "seed": seed,
            "train_samples": len(train_rows),
            "val_samples": len(val_rows),
        },
    )

    train_ds = PreferenceDataset(train_rows)
    val_ds = PreferenceDataset(val_rows)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_list
    )

    # optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)

    best_val_acc = -1.0
    best_path = None
    history = {"train_loss": [], "val_acc": []}
    global_step = 0

    for epoch in range(epochs):
        stats = dpo_train_epoch(
            model=model,
            model_ref=model_ref,
            tokenizer=tokenizer,
            train_loader=train_loader,
            optimizer=optimizer,
            beta=beta,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_grad_norm=max_grad_norm,
            val_ds=val_ds,
            validation_every_steps=validation_every_steps,
            save_every_steps=save_every_steps,
            max_eval_samples=max_eval_samples,
            output_dir=output_dir,
            global_step=global_step,
        )

        global_step = stats["global_step"]
        epoch_best_val_acc = stats.get("best_val_acc", -1.0)
        if epoch_best_val_acc > best_val_acc:
            best_val_acc = epoch_best_val_acc

        history["train_loss"].append(stats["train_loss"])

        # End-of-epoch validation
        val_acc = compute_preference_accuracy(
            model=model,
            tokenizer=tokenizer,
            dataset=val_ds,
            max_eval_samples=max_eval_samples,
        )
        history["val_acc"].append(val_acc)
        print(
            f"Epoch {epoch + 1}/{epochs} - train_loss={stats['train_loss']:.4f} val_acc={val_acc:.4f}"
        )

        # Save best at epoch end if better than step-wise best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(output_dir, exist_ok=True)
            best_path = os.path.join(output_dir, f"best_epoch_{epoch + 1}")
            if hasattr(model, "save_pretrained"):
                model.save_pretrained(best_path)
            if hasattr(tokenizer, "save_pretrained"):
                tokenizer.save_pretrained(best_path)

    return {"best_val_acc": best_val_acc, "best_path": best_path, "history": history}


def get_models_and_tokenizer(model_name: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="cuda:0",
    )
    model_ref = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="cuda:1",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model.config.pad_token_id = tokenizer.pad_token_id
    model_ref.config.pad_token_id = tokenizer.pad_token_id

    model = model.to("cuda:0")
    model.train()
    model_ref = model_ref.to("cuda:1")
    model_ref.eval()
    return model, model_ref, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3.1-8B")
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--val_fraction", type=float, default=0.05)
    parser.add_argument("--max_eval_samples", type=int, default=200)
    args = parser.parse_args()

    model, model_ref, tokenizer = get_models_and_tokenizer(args.model)
    rows = load_anthropic_hh_dpo_dataset()
    train_rows, val_rows = split_train_val(rows, val_fraction=args.val_fraction)

    results = train_dpo(
        model=model,
        model_ref=model_ref,
        tokenizer=tokenizer,
        train_rows=train_rows,
        val_rows=val_rows,
        output_dir=".dpo_checkpoints",
        beta=args.beta,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_eval_samples=args.max_eval_samples,
        validation_every_steps=25,
        save_every_steps=50,
    )
    print(
        f"Best val accuracy: {results['best_val_acc']:.4f} | saved to: {results['best_path']}"
    )


if __name__ == "__main__":
    main()
