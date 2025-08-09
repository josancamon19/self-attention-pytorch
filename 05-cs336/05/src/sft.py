# finetune based on MATH, improve reasoning ability, rather than correct answers, reasoning trace + answer
# /data/a5-alignment/MATH/sft.jsonl (don't have access)
# in practice, sft is used as a warm start for an RL finetuning step, why?
# - sft requires hq annotated data, RL on ly correct answer,
# - RL can most times even with awesome sft data, find better policies
# ~ not for this model sizes, the 2 processes will be treated separately

import torch
from torch import Tensor
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    response_mask = response_masks[:, 1:]

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
    # loss is for optimizing the maximum likelihood estimation, so we do - log likelihood, which is sum of sum(p * logp)
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


def log_generations():
    # Logging generations in-the-loop. It’s always good practice to do some in-the-loop logging that involves
    # generation from your model, and reasoning SFT/RL is no exception. Write a function log_generations
    # that will prompt your model to generate responses for some given prompts (e.g., sampled from the validation
    # set). It’s a good idea to log at least the following for each example:
    # 1. The input prompt.
    # 2. The response generated by the SFT/RL model.
    # 3. The ground-truth answer.
    # 4. The reward information, including format, answer, and total reward.
    # 5. The average token entropy of the response.
    # 6. The average response length, average response length for correct responses, and average response length
    # for incorrect responses.
    pass


# ======== Actual Finetuning using Utils above =========


from vllm.model_executor import set_random_seed as vllm_set_random_seed  # noqa: E402
from vllm import LLM  # noqa: E402
from unittest.mock import patch  # noqa: E402
from transformers import PreTrainedModel, AutoModelForCausalLM, AutoTokenizer  # noqa: E402
import wandb  # noqa: E402
import json  # noqa: E402


def init_vllm(
    model_id: str = "Qwen/Qwen2.5-Math-1.5B",
    device: str = "cuda",
    seed: int = 42,
    gpu_memory_utilization: float = 0.85,
):
    """
    Start the inference process, here we use vLLM to hold a model on a GPU separate from the policy.
    """
    vllm_set_random_seed(seed)
    # Monkeypatch from TRL:
    # https://github.com/huggingface/trl/blob/22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py
    # Patch vLLM to make sure we can
    # (1) place the vLLM model on the desired device (world_size_patch) and
    # (2) avoid a test that is not designed for our setting (profiling_patch).
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None,
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )


def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    """
    Copied from https://github.com/huggingface/trl/blob/22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py#L670.
    """
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


def init_wandb():
    # Setup wandb metrics
    wandb.init("josancamon19", "assignment-05")
    wandb.define_metric("train_step")  # the x‑axis for training
    wandb.define_metric("eval_step")  # the x‑axis for evaluation
    # everything that starts with train/ is tied to train_step
    wandb.define_metric("train/*", step_metric="train_step")
    # everything that starts with eval/ is tied to eval_step
    wandb.define_metric("eval/*", step_metric="eval_step")


def get_model_and_tokenizer(model: str = "Qwen/Qwen2.5-Math-1.5B"):
    model = AutoModelForCausalLM.from_pretrained(
        model,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(model)
    # depends on template
    tokenizer.add_special_tokens(
        {
            "think_start_tag": "<think>",
            "think_closing_tag": "</think>",
            "answer_start_tag": "<answer>",
            "answer_closing_tag": "</answer>",
        }
    )
    return model, tokenizer


def get_dataset_set(
    tokenizer,
    prompt_template_path: str = "src/prompts/r1_zero.prompt",
    dataset: str = "gsm8k",
    subset: str = "train",  # train | test
):
    with open(prompt_template_path, "r") as f:
        prompt_template = f.read().strip()

    with open(f"data/{dataset}/{subset}.jsonl", "r") as f:
        dataset = [json.loads(line.strip()) for line in f]

    prompts, outputs = []
    for item in dataset:
        question = item["question"]
        thinking = item["answer"]
        gt_answer = thinking.split("#### ")[-1].strip()
        prompt = f"{prompt_template.replace('{question}', question)}"
        output = f"{thinking}</think><answer>{gt_answer}</answer>"
        prompts.append(prompt)
        outputs.append(output)
        # could batch this, but meh

    tokenized = tokenize_prompt_and_output(prompts, outputs, tokenizer)
    return (
        tokenized["input_ids"],
        tokenized["labels"],
        tokenized["response_mask"],
    )


# gradient clipping with value 1.0
def run_sft():
    # TODO: why not cross entropy loss here, if we are doing SFT, like just simple finetuning? it's working on dummy.py
    # I guess is different if you are doing a cold start(?) cause the, is the same thing, just more control, need better probability foundations
    init_wandb()
    model, tokenizer = get_model_and_tokenizer()
    # eval_llm = init_vllm()

    model.to(device)

    t_input_ids, t_labels, t_masks = get_dataset_set(tokenizer, subset="train")
    v_input_ids, v_labels, v_masks = get_dataset_set(tokenizer, subset="test")  # val

    batch_size = 16
    learning_rate = 1e-5
    gradient_accumulation_steps = 4
    normalize_constant = 1.0
    # effective batch size = 16 * 4

    optimizer = torch.optim.AdamW(model.parameters(), learning_rate)

    # logging
    train_loss = 0
    avg_token_entropy = 0

    for i in range(0, len(t_input_ids), batch_size):
        j = min(len(t_input_ids), i + batch_size)
        batch_input_ids = t_input_ids[i:j].to(device)
        batch_labels = t_labels[i:j].to(device)
        batch_masks = t_masks[i:j].to(device)

        output = get_response_log_probs(model, batch_input_ids, batch_labels, True)
        log_probs, token_entropy = output["log_probs"], output["token_entropy"]
        loss, metadata = sft_microbatch_train_step(
            log_probs,
            t_masks[i:j],
            gradient_accumulation_steps,
            normalize_constant,
        )
        train_loss += loss.item()
        avg_token_entropy += token_entropy[batch_masks].mean().item()

        if i + 1 % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        if i + 1 % 25 == 0:
            wandb.log(
                {
                    "train/loss": train_loss / i,
                    "train/avg_token_entropy": avg_token_entropy / i,
                    "train_step": i,
                },
                step=i,
            )

        # if i % 500 == 0:
        #     with torch.inference_mode():
        #         load_policy_into_vllm_instance(model, eval_llm)
        #         indices = random.choice()
