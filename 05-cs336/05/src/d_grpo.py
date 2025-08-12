import random
from torch._tensor import Tensor
from typing import Any, Literal
import numpy as np
import wandb
from src.b_sft import (
    EvalDataset,
    get_dataset_set,
    get_model_and_tokenizer,
    get_response_log_probs,
    init_vllm,
    init_wandb,
    load_policy_into_vllm_instance,
    tokenize_prompt_and_output,
)
from src.drgrpo_grader import r1_zero_reward_fn
import torch
from vllm.sampling_params import SamplingParams

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
    # === logging ===
    format_rewards = torch.tensor(format_rewards, device=device).view(-1, group_size)
    answer_rewards = torch.tensor(answer_rewards, device=device).view(-1, group_size)
    # === logging ===

    # the average reward per group, the min and max reward per group
    r_mean = torch.mean(overall_rewards, dim=-1)
    advantages = overall_rewards - r_mean.unsqueeze(-1)
    if normalize_by_std:
        r_std = torch.std(overall_rewards, dim=-1).unsqueeze(-1)
        advantages /= r_std + advantage_eps

    def get_rewards_stats(mean, r):
        return {
            "average": mean.view(-1),
            "min": torch.min(r, -1, keepdim=False),
            "max": torch.max(r, -1, keepdim=False),
            "std": torch.std(r, -1, keepdim=False),
            "percentiles": {
                "25": torch.quantile(r, 0.25, dim=-1),
                "50": torch.quantile(r, 0.5, dim=-1),
                "75": torch.quantile(r, 0.75, dim=-1),
            },
        }

    return (
        advantages.view(-1),
        overall_rewards.view(-1),
        {
            # per group stats
            "overall_rewards": get_rewards_stats(r_mean, overall_rewards),
            "answer_rewards": get_rewards_stats(
                torch.mean(answer_rewards, dim=-1), answer_rewards
            ),
            "format_rewards": get_rewards_stats(
                torch.mean(format_rewards, dim=-1), format_rewards
            ),
        },
    )


def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    # pdb.set_trace()
    return -raw_rewards_or_advantages * policy_log_probs  # .unsqueeze(-1)


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
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if loss_type == "no_baseline":
        loss = compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs)
        return loss, {}
    elif loss_type == "reinforce_with_baseline":
        loss = compute_naive_policy_gradient_loss(advantages, policy_log_probs)
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
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    loss, metadata = compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type=loss_type,
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )

    loss_per_sequence = masked_mean(loss, response_mask)
    scaled_loss = loss_per_sequence / gradient_accumulation_steps
    scaled_loss.backward()
    return scaled_loss, metadata


def train():
    n_grpo_steps: int = 200
    advantage_eps: float = 1e-6
    rollout_batch_size: int = 256
    group_size: int = 8
    sampling_temperature: float = 1.0
    sampling_min_tokens: int = 4  # As in Expiter, disallow empty string responses
    sampling_max_tokens: int = 1024
    sampling_params = SamplingParams(
        temperature=sampling_temperature,
        min_tokens=sampling_min_tokens,
        max_tokens=sampling_max_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
        n=group_size,
    )
    epochs_per_rollout_batch: int = 1  # On-policy
    train_batch_size: int = 256  # On-policy
    gradient_accumulation_steps: int = 128  # microbatch size is 2, will fit on H100
    loss_type: Literal[
        "no_baseline",
        "reinforce_with_baseline",
        "grpo_clip",
    ] = "reinforce_with_baseline"
    use_std_normalization: bool = True

    # Logging frequencies
    log_train_every: int = 1  # Log training metrics every optimizer step
    log_eval_every: int = 10  # Log validation metrics every N steps
    save_checkpoint_every: int = 50  # Save model checkpoints

    # starting hyperparams onpolicy settings
    # for each rollout batch, we take a single gradient step.
    # In terms of hyperparameters, this means that train_batch_size is equal to rollout_ ⌋ batch_size, and epochs_per_rollout_batch is equal to 1.
    assert train_batch_size % gradient_accumulation_steps == 0, (
        "train_batch_size must be divisible by gradient_accumulation_steps"
    )
    micro_train_batch_size = train_batch_size // gradient_accumulation_steps
    assert rollout_batch_size % group_size == 0, (
        "rollout_batch_size must be divisible by group_size"
    )
    n_prompts_per_rollout_batch = rollout_batch_size // group_size
    assert train_batch_size >= group_size, (
        "train_batch_size must be greater than or equal to group_size"
    )
    n_microbatches_per_rollout_batch = rollout_batch_size // micro_train_batch_size
    # -r1_zero prompt, vLLM stop at </answer> tag
    # use typer for argument parsing
    # grad clip 1.0
    # log val rewards every 5/10 steps, vla at least on 1024 examples, as CoT RL evals can be noisy
    # with this implementation, grpo clip should only be used when off policy
    # in the off policy setting with multiple epochs of grad updates per rollout batch, it'd be wasteful to recomp old log probs for each epoch, you can reuse them
    # don't differentiate wrt old log probs
    # should log on every optimizer update (loss, grad norm, token entrpoy, clip fraction if off policy, train rewards, anything else you might think of)

    init_wandb()
    model, tokenizer = get_model_and_tokenizer()

    *_, (t_prompts, t_gt) = get_dataset_set(
        tokenizer, dataset=EvalDataset.MATH, subset="train", return_raw=True
    )
    *_, (v_prompts, v_gt) = get_dataset_set(
        tokenizer, dataset=EvalDataset.MATH, subset="test", return_raw=True
    )

    llm: Any = init_vllm(device="cuda:1", gpu_memory_utilization=0.85)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-5, weight_decay=0.0, betas=(0.9, 0.95)
    )
    indices = [i for i in range(len(t_prompts))]

    # Tracking variables
    total_rollouts = 0
    optimizer_steps = 0

    for i in range(n_grpo_steps):
        batch_indices = random.sample(indices, k=n_prompts_per_rollout_batch)
        load_policy_into_vllm_instance(model, llm)

        batch_prompts = [t_prompts[j] for j in batch_indices]
        batch_gt = [t_gt[j] for j in batch_indices]

        req_outputs = llm.generate(batch_prompts, sampling_params)

        responses = [o.text for ro in req_outputs for o in ro.outputs]
        prompts_rep = [p for p in batch_prompts for _ in range(group_size)]
        gt_rep = [g for g in batch_gt for _ in range(group_size)]

        advantages, raw_rewards, metadata = compute_group_normalized_rewards(
            r1_zero_reward_fn,
            responses,
            gt_rep,
            group_size,
            advantage_eps,
            use_std_normalization,
        )

        tokenized = tokenize_prompt_and_output(prompts_rep, responses, tokenizer)
        batch_input_ids = tokenized["input_ids"].to(device)
        batch_labels = tokenized["labels"].to(device)
        batch_masks = tokenized["response_mask"].to(device)

        # Compute old log probs in microbatches to avoid oom ?
        old_log_probs = []
        with torch.no_grad():
            for j in range(0, batch_input_ids.size(0), micro_train_batch_size):
                bi = batch_input_ids[j : j + micro_train_batch_size]
                bl = batch_labels[j : j + micro_train_batch_size]
                out = get_response_log_probs(model, bi, bl, False)
                old_log_probs.append(out["log_probs"].detach())
            old_log_probs = torch.cat(old_log_probs, dim=0)

        total_rollouts += rollout_batch_size

        for epoch in range(epochs_per_rollout_batch):
            optimizer.zero_grad()

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
                )

                batch_loss += scaled_loss.item() * gradient_accumulation_steps

                # Accumulate metadata if using grpo_clip
                if loss_type == "grpo_clip" and micro_metadata:
                    for key in ["clipped_frac", "ratio_mean", "ratio_std"]:
                        if key in micro_metadata:
                            batch_metadata[key].append(micro_metadata[key])

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer_steps += 1

            # Response length statistics
            response_lengths = batch_masks.sum(dim=-1).float()
            avg_response_length = response_lengths.mean().item()
            success_rate = (raw_rewards > 0).float().mean().item()

            # Log training metrics
            if optimizer_steps % log_train_every == 0:
                log_dict = {
                    "train/loss": batch_loss / n_microbatches_per_rollout_batch,
                    "train/grad_norm": grad_norm.item(),
                    "train/learning_rate": optimizer.param_groups[0]["lr"],
                    # Reward statistics
                    "rewards/mean": raw_rewards.mean().item(),
                    "rewards/std": raw_rewards.std().item(),
                    "rewards/min": raw_rewards.min().item(),
                    "rewards/max": raw_rewards.max().item(),
                    "rewards/success_rate": success_rate,
                    # Advantage statistics
                    "advantages/mean": advantages.mean().item(),
                    "advantages/std": advantages.std().item(),
                    # Response statistics
                    "generation/avg_length": avg_response_length,
                    "generation/min_length": response_lengths.min().item(),
                    "generation/max_length": response_lengths.max().item(),
                    # Detailed reward breakdowns from metadata
                    "rewards/overall_mean": metadata["overall_rewards"]["average"]
                    .mean()
                    .item(),
                    "rewards/answer_mean": metadata["answer_rewards"]["average"]
                    .mean()
                    .item(),
                    "rewards/format_mean": metadata["format_rewards"]["average"]
                    .mean()
                    .item(),
                    # Training progress
                    "train/rollout_step": i,
                    "train/optimizer_step": optimizer_steps,
                    "train/total_rollouts": total_rollouts,
                }

                # Add grpo_clip specific metrics
                if loss_type == "grpo_clip" and batch_metadata["clipped_frac"]:
                    log_dict.update(
                        {
                            "train/clip_fraction": np.mean(
                                batch_metadata["clipped_frac"]
                            ),
                            "train/ratio_mean": np.mean(batch_metadata["ratio_mean"]),
                            "train/ratio_std": np.mean(batch_metadata["ratio_std"]),
                        }
                    )

                wandb.log(log_dict, step=optimizer_steps)

            # Print progress
            if i % 10 == 0:
                avg_reward = raw_rewards.mean().item()
                print(
                    f"Step {i}: avg_reward={avg_reward:.3f}, loss={batch_loss / n_microbatches_per_rollout_batch:.4f}, success_rate={success_rate:.2%}"
                )
        # Validation logging
        if i % log_eval_every == 0 and i > 0:
            with torch.inference_mode():
                # Sample a subset of validation prompts for efficiency
                n_val_samples = min(512, len(v_prompts))
                val_indices = random.sample(range(len(v_prompts)), n_val_samples)
                val_batch_prompts = [v_prompts[j] for j in val_indices]
                val_batch_gt = [v_gt[j] for j in val_indices]

                # Generate validation responses
                load_policy_into_vllm_instance(model, llm)
                val_sampling_params = SamplingParams(
                    temperature=0.1,  # Lower temperature for validation
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
                if i % (log_eval_every * 2) == 0:
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
                    for idx in range(min(5, len(val_responses))):
                        examples_table.add_data(
                            val_batch_prompts[idx],
                            val_responses[idx],
                            val_batch_gt[idx],
                            val_rewards[idx],
                            val_format_rewards[idx],
                            val_answer_rewards[idx],
                        )
                    wandb.log({"eval/examples": examples_table}, step=optimizer_steps)

        # Save checkpoints
        if i % save_checkpoint_every == 0 and i > 0:
            checkpoint_path = f"checkpoints/grpo_step_{i}_opt_{optimizer_steps}"
            model.save_pretrained(checkpoint_path)
            tokenizer.save_pretrained(checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    train()
