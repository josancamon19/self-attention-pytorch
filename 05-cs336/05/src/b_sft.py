# finetune based on MATH, improve reasoning ability, rather than correct answers, reasoning trace + answer
# /data/a5-alignment/MATH/sft.jsonl (don't have access)
# in practice, sft is used as a warm start for an RL finetuning step, why?
# - sft requires hq annotated data, RL on ly correct answer,
# - RL can most times even with awesome sft data, find better policies
# ~ not for this model sizes, the 2 processes will be treated separately

import torch
import random
import enum
from torch import Tensor
from tqdm import tqdm
from vllm.sampling_params import SamplingParams
from src.a_eval import evaluate_model, compute_eval_stats
import os
from pathlib import Path

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.makedirs(".checkpoints", exist_ok=True)


class EvalDataset(enum.Enum):
    GSM8K = "gsm8k"
    MATH = "math"


datasets = {"gsm8k": "data/gsm8k/test.jsonl", "math": "data/math/validation.jsonl"}


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
    # Build attention mask from model pad token id if available
    # Handle torch.compile wrapped models by accessing the original model's config
    if hasattr(model, "_orig_mod"):
        config = model._orig_mod.config
    else:
        config = model.config

    pad_id = getattr(config, "pad_token_id", None)
    if pad_id is not None:
        # one thing is the loss padding, and the attention padding, as I need static shapes for batching
        # padding was added manually in the encode func above, thus we need to retrieve that attn mask now, to avoid computing attn
        # on dumb tokens
        attention_mask = (input_ids != pad_id).to(input_ids.device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    else:
        logits = model(input_ids=input_ids).logits
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
    loss_per_sequence = -masked_normalize(
        policy_log_probs, response_mask, dim=-1, normalize_constant=normalize_constant
    )  # [-4.4235,  2.0203]
    # non padding tokens per sequence

    # Correct ? ❌
    # # how much of the loss to attribute to each sequence given how many non padded tokens it has # [0.6, 0.8]
    # tokens_per_sequence = response_mask.sum(dim=-1).clamp_min(1) # 6,8
    # valid_tokens_ratio_per_seq = tokens_per_sequence / policy_log_probs.shape[-1]
    # # correctly attribute loss per sequence
    # loss = loss_per_sequence * valid_tokens_ratio_per_seq # [-2.6541,  1.6163]
    # loss = loss.mean()  # avg loss per sequence # -0.5189
    # pdb.set_trace()
    # NVM, loss of a microbatch is a lost per sequence avg, we are already ignoring padding on loss_per_sequence, thus this ratio doesn't make sense

    # Passes the test ✅ but is wrong? = this one means we are attributing loss to padding tokens as well?
    loss = loss_per_sequence.mean()  # avg loss per sequence # -1.2016

    # we'd do this same at batch level, so this portion is 1/nth of total loss of a batch
    scaled_loss = loss / gradient_accumulation_steps
    scaled_loss.backward()
    metadata = {
        "avg_loss_per_sequence": loss.item(),
        "num_masked_tokens": response_mask.sum().item(),
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
    device: str = "cuda:1",
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
    # Handle torch.compile wrapped models by accessing the original model
    if hasattr(policy, "_orig_mod"):
        # For torch.compile wrapped models, get state_dict from the original model
        state_dict = policy._orig_mod.state_dict()
    else:
        # For non-compiled models
        state_dict = policy.state_dict()

    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


def init_wandb():
    # Setup wandb metrics
    wandb.init("josancamon19-cifrato", "assignment-05")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")


def get_model_and_tokenizer(model_name: str = "Qwen/Qwen2.5-Math-1.5B"):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="cuda:0",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.config.pad_token_id = tokenizer.pad_token_id
    # depends on template
    # tokenizer.add_tokens(["<think>", "</think>", "<answer>", "</answer>"])
    # model.resize_token_embeddings(len(tokenizer))
    model = model.to(device)
    model.train()
    # Compile the model with specific options to avoid NameError issues
    # model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
    return model, tokenizer


def get_dataset_set(
    tokenizer,
    prompt_template_path: str = "src/prompts/r1_zero.prompt",
    dataset: EvalDataset = EvalDataset.GSM8K,
    subset: str = "train",  # train | test | sft (train)
    return_raw: bool = False,
):
    # Resolve paths relative to the repository root to work under Ray trials
    repo_root = Path(__file__).resolve().parent.parent
    dataset_path = repo_root / f"data/{dataset.value}/{subset}.jsonl"
    if not Path(prompt_template_path).is_absolute():
        prompt_template_path = str(repo_root / prompt_template_path)

    with open(dataset_path, "r") as f:
        dataset_items = [json.loads(line.strip()) for line in f]

    prompts, outputs, ground_truths = [], [], []

    with open(prompt_template_path, "r") as f:
        prompt_template = f.read().strip()

    for item in dataset_items:
        if dataset == EvalDataset.MATH:
            if subset == "sft":
                if len(item["prompt"]) + len(item["response"]) > 2000:
                    continue  # OOM
                prompts.append(item["prompt"])
                outputs.append(item["response"])
                ground_truths.append(item["ground_truth"])
            else:  # train / test
                prompt = prompt_template.replace("{question}", item["problem"])
                output = (
                    f"nothing here matters</think><answer>{item['answer']}</answer>"
                )
                if len(prompt) + len(output) > 2000:
                    continue  # OOM
                prompts.append(prompt)
                outputs.append(output)
                ground_truths.append(item["answer"])
        else:  # GSM8K
            assert prompt_template == "src/prompts/r1_zero.prompt"

            gt_answer = item["answer"].split("#### ")[-1].strip()

            prompt = prompt_template.replace("{question}", item["question"])
            output = f"{item['answer']}</think><answer>{gt_answer}</answer>"
            prompts.append(prompt)
            outputs.append(output)
            ground_truths.append(gt_answer)

    # avg_prompt_len = sum(len(p) for p in prompts) / len(prompts) if prompts else 0
    # avg_output_len = sum(len(o) for o in outputs) / len(outputs) if outputs else 0
    # max_prompt_len = max((len(p) for p in prompts), default=0)
    # max_output_len = max((len(o) for o in outputs), default=0)
    # max_total_len = max((len(p) + len(o) for p, o in zip(prompts, outputs)), default=0)
    # over_2000 = sum(1 for p, o in zip(prompts, outputs) if len(p) + len(o) > 2000)
    # print(f"Average prompt length: {avg_prompt_len:.2f}, max: {max_prompt_len}")
    # print(f"Average output length: {avg_output_len:.2f}, max: {max_output_len}")
    # print(f"Max prompt+output length: {max_total_len}")
    # print(f"Number with prompt+output length > 2000: {over_2000}")
    # print(f"Number of items: {len(dataset_items)}")
    tokenized = tokenize_prompt_and_output(prompts, outputs, tokenizer)
    output = [
        tokenized["input_ids"],
        tokenized["labels"],
        tokenized["response_mask"],
    ]
    if return_raw:
        output.append((prompts, ground_truths))
    return output


def compute_val_loss(model, v_input_ids, v_labels, v_masks, micro_batch_size=16):
    num = 0.0
    den = 0
    with torch.inference_mode():
        for j in range(0, v_input_ids.size(0), micro_batch_size):
            bi = v_input_ids[j : j + micro_batch_size].to(device)
            bl = v_labels[j : j + micro_batch_size].to(device)
            bm = v_masks[j : j + micro_batch_size].to(device)

            out = get_response_log_probs(model, bi, bl, return_token_entropy=False)
            log_probs = out["log_probs"]  # [B, T]
            num += (-(log_probs * bm)).sum().item()  # total NLL over masked tokens
            den += bm.sum().item()  # total masked tokens

    return num / max(1, den)  # average NLL per masked token


def compute_reasoning_val_loss(
    llm,
    prompts,
    ground_truths,
    temperature: float = 1.0,
    top_p: float = 1,
):
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=512,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )
    indices = set(random.choices([i for i in range(len(prompts))], k=len(prompts)))
    results = evaluate_model(
        llm,
        sampling_params,
        [p for i, p in enumerate(prompts) if i in indices],
        [gt for i, gt in enumerate(ground_truths) if i in indices],
    )
    return compute_eval_stats(results, False)


def run_sft(dataset: EvalDataset = EvalDataset.GSM8K):
    init_wandb()
    model, tokenizer = get_model_and_tokenizer()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    t_input_ids, t_labels, t_masks = get_dataset_set(
        tokenizer, dataset=dataset, subset="sft" if dataset.value == "math" else "train"
    )
    v_input_ids, v_labels, v_masks, (val_prompts, val_gt) = get_dataset_set(
        tokenizer, dataset=dataset, subset="test", return_raw=True
    )
    eval_llm = init_vllm()
    micro_batch_size = 16
    gradient_accumulation_steps = 4
    # effective_batch_size = micro_batch_size * gradient_accumulation_steps # 16*4 = 64

    # logging
    batch_loss = 0
    batch_entropy = 0
    train_loss = 0
    optimizer_steps = 0

    # epoch/step handling
    epochs = 5
    microbatches_per_epoch = (
        len(t_input_ids) + micro_batch_size - 1
    ) // micro_batch_size
    total_microbatches = microbatches_per_epoch * epochs
    indices = [i for i in range(len(t_input_ids))]

    with tqdm(total=total_microbatches, dynamic_ncols=True) as pbar:
        for i in range(total_microbatches):
            current_epoch = (i // microbatches_per_epoch) + 1
            ga_step = (i % gradient_accumulation_steps) + 1
            pbar.set_description(f"Epoch {current_epoch}/{epochs}")

            if i % microbatches_per_epoch == 0:
                random.shuffle(indices)
            # Per-epoch slice
            epoch_i = i % microbatches_per_epoch
            start = epoch_i * micro_batch_size
            end = min(start + micro_batch_size, len(indices))
            if end <= start:
                pbar.update(1)
                continue
            batch_indices = indices[start:end]
            batch_input_ids = t_input_ids[batch_indices].to(device)
            batch_labels = t_labels[batch_indices].to(device)
            batch_masks = t_masks[batch_indices].to(device)

            output = get_response_log_probs(model, batch_input_ids, batch_labels, True)
            log_probs, token_entropy = output["log_probs"], output["token_entropy"]
            loss, metadata = sft_microbatch_train_step(
                log_probs, batch_masks, gradient_accumulation_steps
            )
            batch_loss += loss.item()
            batch_entropy += token_entropy[batch_masks].mean().item()

            if (i + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

                optimizer_steps += 1
                train_loss += batch_loss

                wandb.log(
                    {
                        "train/batch_loss": batch_loss,
                        "train/loss": train_loss / optimizer_steps,
                        "train/avg_token_entropy": batch_entropy
                        / gradient_accumulation_steps,
                        "train_step": optimizer_steps,
                    },
                    step=optimizer_steps,
                )
                batch_loss = 0.0
                batch_entropy = 0.0

            avg_train_loss = (
                (train_loss / optimizer_steps) if optimizer_steps > 0 else 0.0
            )
            pbar.set_postfix(
                train_loss=f"{avg_train_loss:.4f}",
                epoch=current_epoch,
                ga=f"{ga_step}/{gradient_accumulation_steps}",
                step=optimizer_steps,
            )
            pbar.update(1)

            # when i == 0, both will trigger just to understand the baseline
            # no reasoning trace on math, so this would be meaningless
            if dataset != EvalDataset.MATH and i % 100 == 0:
                val_loss = compute_val_loss(
                    model, v_input_ids, v_labels, v_masks, micro_batch_size
                )
                wandb.log(
                    {"eval/loss": val_loss, "eval_step": optimizer_steps},
                    step=optimizer_steps,
                )

            if i % 100 == 0:
                with torch.inference_mode():
                    load_policy_into_vllm_instance(model, eval_llm)
                    format_accuracy, answer_accuracy, overall_accuracy = (
                        compute_reasoning_val_loss(eval_llm, val_prompts, val_gt)
                    )
                    wandb.log(
                        {
                            "eval/format_accuracy": format_accuracy,
                            "eval/answer_accuracy": answer_accuracy,
                            "eval/overall_accuracy": overall_accuracy,
                            "eval_step": optimizer_steps,
                        },
                        step=optimizer_steps,
                    )
            if i % 100 == 0 and i > 0:  # Save every 500 steps
                model.save_pretrained(f".checkpoints/step_{optimizer_steps}")
                tokenizer.save_pretrained(f".checkpoints/step_{optimizer_steps}")


if __name__ == "__main__":
    run_sft(EvalDataset.MATH)
    # TODO: learning rate tuning, and batch size, how much memory is being used, cos lr schedule
    # TODO: saving the model
    # TODO: making a full run, can you get >20%? results ~ what about loss per token, how that affects
    # TODO: how was this 2.5B model trained? hasn't it seen all math datasets already?
