import random
from torch._tensor import Tensor
from typing import Any, Literal
import numpy as np
from transformers import get_cosine_schedule_with_warmup
import wandb
from src.b_sft import (
    EvalDataset,
    get_dataset_set,
    get_model_and_tokenizer,
    get_response_log_probs,
    init_vllm,
    load_policy_into_vllm_instance,
    tokenize_prompt_and_output,
)
from src.drgrpo_grader import r1_zero_reward_fn
import torch
from vllm.sampling_params import SamplingParams
import pdb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# clipping = stability when taking many gradient steps on a single batch of rollouts
def compute_group_normalized_rewards(
    reward_fn: callable,
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    rewards = [
        reward_fn(a, b)  # reward, format_reward, answer_reward
        for a, b in zip(rollout_responses, repeated_ground_truths)
    ]
    overall_rewards = [item["reward"] for item in rewards]
    format_rewards = [item["format_reward"] for item in rewards]
    answer_rewards = [item["answer_reward"] for item in rewards]

    overall_rewards = torch.tensor(overall_rewards, device=device).view(-1, group_size)

    # the average reward per group, the min and max reward per group
    r_mean = torch.mean(overall_rewards, dim=-1)
    advantages = overall_rewards - r_mean.unsqueeze(-1)
    if normalize_by_std:
        r_std = torch.std(overall_rewards, dim=-1).unsqueeze(-1)
        advantages /= r_std + advantage_eps

    return (
        advantages.view(-1),
        overall_rewards.view(-1),
        {
            # per group stats
            "answer_rewards": torch.tensor(answer_rewards, device=device),
            "format_rewards": torch.tensor(format_rewards, device=device),
        },
    )

def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    unsqueeze_fix: bool = False
) -> torch.Tensor:
    if unsqueeze_fix:
        # the test has no batch_size, so when batch size != 1, training fails
        return -raw_rewards_or_advantages.unsqueeze(-1) * policy_log_probs  
    return -raw_rewards_or_advantages * policy_log_probs  


def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    ratio: Tensor = torch.exp(policy_log_probs - old_log_probs)
    clipped_ratio = torch.clamp(ratio, 1 - cliprange, 1 + cliprange)
    unclipped_loss = ratio * advantages
    clipped_loss = clipped_ratio * advantages
    output = -torch.minimum(unclipped_loss, clipped_loss)

    # Compute relevant stats
    clipped_mask = ratio != clipped_ratio
    stats = {
        "ratio_mean": ratio.mean().item(),
        "ratio_std": ratio.std().item(),
        "clipped_frac": clipped_mask.float().mean().item(),
        "num_clipped": clipped_mask.sum().item(),
        "loss_mean": output.mean().item(),
        "loss_std": output.std().item(),
    }
    return output, stats


def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    unsqueeze_fix : bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if loss_type == "no_baseline":
        loss = compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs,unsqueeze_fix)
        return loss, {}
    elif loss_type == "reinforce_with_baseline":
        loss = compute_naive_policy_gradient_loss(advantages, policy_log_probs,unsqueeze_fix)
        return loss, {}
    elif loss_type == "grpo_clip":
        return compute_grpo_clip_loss(
            advantages=advantages,
            policy_log_probs=policy_log_probs,
            old_log_probs=old_log_probs,
            cliprange=cliprange,
        )


def masked_mean(
    tensor: torch.Tensor, mask: torch.Tensor, dim: int | None = None
) -> torch.Tensor:
    masked = tensor * mask
    total = mask.sum(dim=dim)
    return masked.sum(dim=dim) / total


def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    unsqueeze_fix: bool = False
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    loss, metadata = compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type=loss_type,
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
        unsqueeze_fix=unsqueeze_fix,
    )

    loss_per_sequence = masked_mean(loss, response_mask)
    scaled_loss = loss_per_sequence / gradient_accumulation_steps
    scaled_loss.backward()
    return scaled_loss, metadata


# ==============================
# ======= GRPO Training ========
# ==============================


def log_step(
    i,
    optimizer_steps,
    log_train_every,
    loss_type,
    batch_masks,
    batch_loss,
    grad_norm,
    optimizer,
    n_microbatches_per_rollout_batch,
    raw_rewards,
    advantages,
    metadata,
    batch_metadata,
):
    if not (optimizer_steps % log_train_every == 0):
        return

    response_lengths = batch_masks.sum(dim=-1).float()
    avg_response_length = response_lengths.mean().item()
    success_rate = (raw_rewards > 0).float().mean().item()

    log_dict = {
        "train/loss": batch_loss / n_microbatches_per_rollout_batch,
        "train/grad_norm": grad_norm.item(),
        "train/learning_rate": optimizer.param_groups[0]["lr"],
        # Reward statistics
        "rewards/mean": raw_rewards.mean().item(),
        "rewards/std": raw_rewards.std().item(),
        "rewards/min": raw_rewards.min().item(),
        "rewards/max": raw_rewards.max().item(),
        "rewards/answer_mean": metadata["answer_rewards"].mean().item(),
        "rewards/format_mean": metadata["format_rewards"].mean().item(),
        "rewards/success_rate": success_rate,
        # Advantage statistics
        "advantages/mean": advantages.mean().item(),
        "advantages/std": advantages.std().item(),
        # Response statistics
        "generation/avg_length": avg_response_length,
        "generation/min_length": response_lengths.min().item(),
        "generation/max_length": response_lengths.max().item(),
    }

    # Add grpo_clip specific metrics
    if loss_type == "grpo_clip" and batch_metadata["clipped_frac"]:
        log_dict.update(
            {
                "train/clip_fraction": np.mean(batch_metadata["clipped_frac"]),
                "train/ratio_mean": np.mean(batch_metadata["ratio_mean"]),
                "train/ratio_std": np.mean(batch_metadata["ratio_std"]),
            }
        )

    wandb.log(log_dict, step=optimizer_steps)
    print(
        f"Step {i}: avg_reward={raw_rewards.mean().item():.3f}, loss={batch_loss / n_microbatches_per_rollout_batch:.4f}, success_rate={success_rate:.2%}"
    )


def run_validation(
    i,
    optimizer_steps,
    log_eval_every,
    v_prompts,
    v_gt,
    model,
    llm,
    sampling_min_tokens,
    sampling_max_tokens,
    sampling_temperature,
):
    if i % log_eval_every == 0 and i > 0:
        with torch.inference_mode():
            # Sample a subset of validation prompts for efficiency
            n_val_samples = min(1024, len(v_prompts))
            val_indices = random.sample(range(len(v_prompts)), n_val_samples)
            val_batch_prompts = [v_prompts[j] for j in val_indices]
            val_batch_gt = [v_gt[j] for j in val_indices]

            # Generate validation responses
            load_policy_into_vllm_instance(model, llm)
            val_sampling_params = SamplingParams(
                temperature=sampling_temperature,  # Lower temperature for validation
                min_tokens=sampling_min_tokens,
                max_tokens=sampling_max_tokens,
                stop=["</answer>"],
                include_stop_str_in_output=True,
                n=1,  # Single response for validation
            )
            val_outputs = llm.generate(val_batch_prompts, val_sampling_params)
            val_responses = [o.outputs[0].text for o in val_outputs]

            # Compute validation rewards
            val_rewards = []
            val_format_rewards = []
            val_answer_rewards = []
            for resp, gt in zip(val_responses, val_batch_gt):
                reward_dict = r1_zero_reward_fn(resp, gt)
                val_rewards.append(reward_dict["reward"])
                val_format_rewards.append(reward_dict["format_reward"])
                val_answer_rewards.append(reward_dict["answer_reward"])

            # Log validation metrics
            val_log_dict = {
                "eval/reward_mean": np.mean(val_rewards),
                "eval/reward_std": np.std(val_rewards),
                "eval/format_accuracy": np.mean(val_format_rewards),
                "eval/answer_accuracy": np.mean(val_answer_rewards),
                "eval/overall_accuracy": np.mean([r > 0 for r in val_rewards]),
                "eval/n_samples": n_val_samples,
            }
            wandb.log(val_log_dict, step=optimizer_steps)

            # Log a few example generations
            examples_table = wandb.Table(
                columns=[
                    "prompt",
                    "response",
                    "ground_truth",
                    "reward",
                    "format_reward",
                    "answer_reward",
                ]
            )
            
            print(f"\n{'='*80}")
            print(f"VALIDATION EXAMPLES - Step {i} (Optimizer Step {optimizer_steps})")
            print(f"{'='*80}")
            
            for idx in range(min(5, len(val_responses))):
                # Convert any tensor values to Python types for wandb
                reward_val = float(val_rewards[idx]) if hasattr(val_rewards[idx], 'item') else val_rewards[idx]
                format_reward_val = float(val_format_rewards[idx]) if hasattr(val_format_rewards[idx], 'item') else val_format_rewards[idx]
                answer_reward_val = float(val_answer_rewards[idx]) if hasattr(val_answer_rewards[idx], 'item') else val_answer_rewards[idx]
                
                examples_table.add_data(
                    val_batch_prompts[idx],
                    val_responses[idx],
                    val_batch_gt[idx],
                    reward_val,
                    format_reward_val,
                    answer_reward_val,
                )
                
                # Console logging for each example
                print(f"\n--- Example {idx + 1} ---")
                print(f"Prompt: {val_batch_prompts[idx][:200]}{'...' if len(val_batch_prompts[idx]) > 200 else ''}")
                print(f"Response: {val_responses[idx][:300]}{'...' if len(val_responses[idx]) > 300 else ''}")
                print(f"Ground Truth: {val_batch_gt[idx][:100]}{'...' if len(val_batch_gt[idx]) > 100 else ''}")
                print(f"Rewards → Total: {reward_val:.3f}, Format: {format_reward_val:.3f}, Answer: {answer_reward_val:.3f}")
            
            print(f"{'='*80}\n")
            wandb.log({"eval/examples": examples_table}, step=optimizer_steps)


def compute_old_log_probs(
    old_policy, batch_input_ids, batch_labels, micro_train_batch_size
):
    old_log_probs = []
    with torch.no_grad():
        for j in range(0, batch_input_ids.size(0), micro_train_batch_size):
            bi = batch_input_ids[j : j + micro_train_batch_size]
            bl = batch_labels[j : j + micro_train_batch_size]
            out = get_response_log_probs(old_policy, bi, bl, False)
            old_log_probs.append(out["log_probs"].detach())
        old_log_probs = torch.cat(old_log_probs, dim=0)
    return old_log_probs


def process_batch(
    batch_input_ids,
    batch_labels,
    batch_masks,
    micro_train_batch_size,
    model,
    raw_rewards,
    advantages,
    old_log_probs,
    gradient_accumulation_steps,
    loss_type,
):
    # Tracking metrics for this batch
    batch_loss = 0.0
    batch_metadata = {"clipped_frac": [], "ratio_mean": [], "ratio_std": []}

    # Process in microbatches for gradient accumulation
    for j in range(0, batch_input_ids.size(0), micro_train_batch_size):
        bi = batch_input_ids[j : j + micro_train_batch_size]
        bl = batch_labels[j : j + micro_train_batch_size]
        bm = batch_masks[j : j + micro_train_batch_size]

        output = get_response_log_probs(model, bi, bl, False)
        log_probs = output["log_probs"]

        # Slice rewards/advantages for this microbatch
        micro_rewards = raw_rewards[j : j + micro_train_batch_size]
        micro_advantages = advantages[j : j + micro_train_batch_size]
        micro_old_log_probs = old_log_probs[j : j + micro_train_batch_size]

        scaled_loss, micro_metadata = grpo_microbatch_train_step(
            log_probs,
            bm,
            gradient_accumulation_steps,
            loss_type,
            micro_rewards,
            micro_advantages,
            micro_old_log_probs,
            0.2,
            True,
        )

        batch_loss += scaled_loss.item() * gradient_accumulation_steps

        # Accumulate metadata if using grpo_clip
        if loss_type == "grpo_clip" and micro_metadata:
            for key in ["clipped_frac", "ratio_mean", "ratio_std"]:
                if key in micro_metadata:
                    batch_metadata[key].append(micro_metadata[key])
    return batch_loss, batch_metadata


def obtain_step_grouped_rollouts(
    indices,
    n_prompts_per_rollout_batch,
    llm,
    sampling_temperature,
    sampling_min_tokens,
    sampling_max_tokens,
    group_size,
    prompts,
    ground_truths,
):
    sampling_params = SamplingParams(
        temperature=sampling_temperature,
        min_tokens=sampling_min_tokens,
        max_tokens=sampling_max_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
        n=group_size,
    )
    batch_indices = random.sample(indices, k=n_prompts_per_rollout_batch)
    batch_prompts = [prompts[j] for j in batch_indices]
    req_outputs = llm.generate(batch_prompts, sampling_params)
    responses = [o.text for ro in req_outputs for o in ro.outputs]

    # just repeating data, util variables
    batch_gt = [ground_truths[j] for j in batch_indices]
    prompts_rep = [p for p in batch_prompts for _ in range(group_size)]
    gt_rep = [g for g in batch_gt for _ in range(group_size)]
    return responses, prompts_rep, gt_rep


def train(
    n_grpo_steps: int = 200,
    rollout_batch_size: int = 256,
    group_size: int = 8,
    sampling_temperature: float = 1.0,
    sampling_min_tokens: int = 4,
    sampling_max_tokens: int = 1024,
    epochs_per_rollout_batch: int = 1,  # On-policy
    train_batch_size: int = 256,  # On-policy
    loss_type: str = "reinforce_with_baseline",  # "no_baseline", "reinforce_with_baseline" "grpo_clip"
    use_std_normalization: bool = True,
    max_lr: float = 1e-5,
):
    num_gpus = torch.cuda.device_count()
    print(f"Detected {num_gpus} GPU(s)")
    gradient_accumulation_steps: int = 128

    if num_gpus == 1:
        vllm_device = "cuda:0"
        vllm_gpu_memory_utilization = 0.6
        # Reduce micro batch size and increase gradient accumulation to maintain effective batch size
        adjusted_gradient_accumulation_steps = gradient_accumulation_steps * 2 # 1 micro
        print(
            f"Adjusted gradient_accumulation_steps: {gradient_accumulation_steps} -> {adjusted_gradient_accumulation_steps}"
        )
    else:
        vllm_device = "cuda:1"
        vllm_gpu_memory_utilization = 0.85
        adjusted_gradient_accumulation_steps = gradient_accumulation_steps // 4 # I think can go higher on this only 27GB
        print("Multi-GPU detected: Using standard settings")

    # Logging frequencies
    log_train_every: int = 1  # Log training metrics every optimizer step
    log_eval_every: int = 5  # Log validation metrics every N steps
    save_checkpoint_every: int = 50  # Save model checkpoints

    assert train_batch_size % adjusted_gradient_accumulation_steps == 0, (
        "train_batch_size must be divisible by adjusted_gradient_accumulation_steps"
    )
    micro_train_batch_size = train_batch_size // adjusted_gradient_accumulation_steps
    assert rollout_batch_size % group_size == 0, (
        "rollout_batch_size must be divisible by group_size"
    )
    n_prompts_per_rollout_batch = rollout_batch_size // group_size
    assert train_batch_size >= group_size, (
        "train_batch_size must be greater than or equal to group_size"
    )
    n_microbatches_per_rollout_batch = rollout_batch_size // micro_train_batch_size
    # use typer for argument parsing
    # with this implementation, grpo clip should only be used when off policy
    # in the off policy setting with multiple epochs of grad updates per rollout batch, it'd be wasteful to recomp old log probs for each epoch, you can reuse them

    wandb.init(project="assignment-05-grpo", config=locals())
    model, tokenizer = get_model_and_tokenizer()

    *_, (t_prompts, t_gt) = get_dataset_set(
        tokenizer, dataset=EvalDataset.MATH, subset="train", return_raw=True
    )
    *_, (v_prompts, v_gt) = get_dataset_set(
        tokenizer, dataset=EvalDataset.MATH, subset="test", return_raw=True
    )

    llm: Any = init_vllm(
        device=vllm_device, gpu_memory_utilization=vllm_gpu_memory_utilization
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=max_lr, weight_decay=0.0, betas=(0.9, 0.95)
    )
    num_optimizer_steps = n_grpo_steps * epochs_per_rollout_batch
    warmup_steps = max(50, int(0.05 * num_optimizer_steps))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, warmup_steps, num_optimizer_steps
    )

    indices = [i for i in range(len(t_prompts))]

    optimizer_steps = 0

    for i in range(n_grpo_steps):
        load_policy_into_vllm_instance(model, llm)  # old_policy

        responses, prompts_rep, gt_rep = obtain_step_grouped_rollouts(
            indices,
            n_prompts_per_rollout_batch,
            llm,
            sampling_temperature,
            sampling_min_tokens,
            sampling_max_tokens,
            group_size,
            t_prompts,
            t_gt,
        )

        advantages, raw_rewards, metadata = compute_group_normalized_rewards(
            r1_zero_reward_fn,
            responses,
            gt_rep,
            group_size,
            1e-6,
            use_std_normalization,
        )

        tokenized = tokenize_prompt_and_output(prompts_rep, responses, tokenizer)
        batch_input_ids = tokenized["input_ids"].to(device)
        batch_labels = tokenized["labels"].to(device)
        batch_masks = tokenized["response_mask"].to(device)

        # needed for the ratio in grpo clip only
        old_log_probs = compute_old_log_probs(
            model,
            batch_input_ids,
            batch_labels,
            micro_train_batch_size,
        )

        for epoch in range(epochs_per_rollout_batch):
            optimizer.zero_grad()

            batch_loss, batch_metadata = process_batch(
                batch_input_ids,
                batch_labels,
                batch_masks,
                micro_train_batch_size,
                model,
                raw_rewards,
                advantages,
                old_log_probs,
                adjusted_gradient_accumulation_steps,
                loss_type,
            )

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer_steps += 1

            log_step(
                i,
                optimizer_steps,
                log_train_every,
                loss_type,
                batch_masks,
                batch_loss,
                grad_norm,
                optimizer,
                n_microbatches_per_rollout_batch,
                raw_rewards,
                advantages,
                metadata,
                batch_metadata,
            )

        run_validation(
            i,
            optimizer_steps,
            log_eval_every,
            v_prompts,
            v_gt,
            model,
            llm,
            sampling_min_tokens,
            sampling_max_tokens,
            sampling_temperature,
        )

        # Save checkpoints
        if i % save_checkpoint_every == 0 and i > 0:
            checkpoint_path = f".checkpoints/grpo_step_{i}_opt_{optimizer_steps}"
            model.save_pretrained(checkpoint_path)
            tokenizer.save_pretrained(checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    import typer
    
    typer.run(train)
