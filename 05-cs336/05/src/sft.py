# finetune based on MATH, improve reasoning ability, rather than correct answers, reasoning trace + answer
# /data/a5-alignment/MATH/sft.jsonl (don't have access)
# in practice, sft is used as a warm start for an RL finetuning step, why?
# - sft requires hq annotated data, RL on ly correct answer,
# - RL can most times even with awesome sft data, find better policies
# ~ not for this model sizes, the 2 processes will be treated separately

import torch
from torch import Tensor


# This could prob be more efficient
def tokenize_prompt_and_output(
    prompt_strs: list[str], output_strs: list[str], tokenizer
):
    p_encoded = tokenizer(prompt_strs, padding=False, truncation=False)["input_ids"]
    out_encoded = tokenizer(output_strs, padding=False, truncation=False)["input_ids"]

    joined: list[tuple[Tensor, Tensor]] = [
        (torch.tensor(p), torch.tensor(out)) for p, out in zip(p_encoded, out_encoded)
    ]
    lengths = [(len(p), len(out)) for p, out in joined]
    max_length = max(p + out for p, out in lengths)

    masks = [
        torch.cat(
            [
                torch.zeros(p_length, dtype=torch.bool),
                torch.ones(out_length, dtype=torch.bool),
                torch.zeros(max_length - (p_length + out_length), dtype=torch.bool),
            ]
        )
        for p_length, out_length in lengths
    ]
    # Concatenate and pad sequences
    concatenated = []
    for a, b in joined:
        # Concatenate prompt + output
        concat_seq = torch.cat([a, b])
        # Pad to max_length
        padding_len = max_length - len(concat_seq)
        if padding_len > 0:
            padded_seq = torch.cat(
                [concat_seq, torch.full((padding_len,), tokenizer.pad_token_id)]
            )
        else:
            padded_seq = concat_seq
        concatenated.append(padded_seq)

    # Stack into tensors
    full_sequences = torch.stack(concatenated)
    response_masks = torch.stack(masks)

    # Create final tensors with correct shapes
    input_ids = full_sequences[:, :-1]
    labels = full_sequences[:, 1:]
    response_mask = response_masks[:, 1:]  # Shift mask to align with labels

    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask,
    }


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    # Logging per-token entropies. When doing RL, it is often useful to keep track of per-token entropies to
    # see if the predictive distribution of the model is becoming (over)confident. We will implement this now and
    # compare how each of our finetuning approaches affects the model's predictive entropy.

    # entropy is the expected surprise, surprise, is the inverse of prob, which is the log, and prob is softmax of logits
    log_p = torch.log_softmax(logits, dim=-1)
    p = torch.exp(log_p)
    return -torch.sum(p * log_p, dim=-1)


def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    logits = model(input_ids).logits
    out_log_probs = torch.log_softmax(logits, dim=-1)
    log_probs = torch.gather(out_log_probs, dim=-1, index=labels.unsqueeze(-1))

    output = {"log_probs": log_probs.squeeze(-1)}
    if return_token_entropy:
        output["token_entropy"] = compute_entropy(logits)
    return output


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
    normalize_constant: float = 1.0,
):
    # sums over tensor elements and normalizes by a constant while respecting a boolean mask.
    return torch.sum(tensor * mask, dim=dim) / normalize_constant


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: int | None = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    loss = -masked_normalize(
        policy_log_probs, response_mask, dim=-1, normalize_constant=normalize_constant
    )

    loss = loss.mean()
    scaled_loss = loss / gradient_accumulation_steps
    scaled_loss.backward()
    metadata = {
        "loss": loss.item(),  # Unscaled loss for logging
        "num_masked_tokens": response_mask.sum().item(),
        "sequence_length": policy_log_probs.shape[-1],
        "batch_size": policy_log_probs.shape[0] if policy_log_probs.dim() > 1 else 1,
    }
    return (scaled_loss, metadata)
