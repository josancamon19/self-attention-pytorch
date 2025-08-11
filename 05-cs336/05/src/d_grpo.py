import pdb
import random
from typing import Any, Literal
from src.b_sft import (
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
        reward_fn(a, b)["reward"]  # reward, format_reward, answer_reward
        for a, b in zip(rollout_responses, repeated_ground_truths)
    ]
    rewards = torch.tensor(rewards, device=device).view(-1, group_size)
    r_mean = torch.mean(rewards, dim=-1)
    advantages = rewards - r_mean.unsqueeze(-1)
    if normalize_by_std:
        r_std = torch.std(rewards, dim=-1).unsqueeze(-1)
        advantages /= r_std + advantage_eps

    return advantages.view(-1), rewards.view(-1), {}


def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    # pdb.set_trace()
    return -raw_rewards_or_advantages.unsqueeze(-1) * policy_log_probs # .unsqueeze(-1)


def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    ratio = torch.exp(policy_log_probs - old_log_probs)
    clipped_ratio = torch.clamp(ratio, 1 - cliprange, 1 + cliprange)
    output = -torch.minimum(ratio * advantages, clipped_ratio * advantages)
    return output, {}


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
    return scaled_loss, {}


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

    # move to experiments (8th section afterwards)
    init_wandb()
    model, tokenizer = get_model_and_tokenizer()
    llm: Any = init_vllm(device="cuda:1", gpu_memory_utilization=0.85)

    *_, (t_prompts, t_gt) = get_dataset_set(tokenizer, subset="train", return_raw=True)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-5, weight_decay=0.0, betas=(0.9, 0.95)
    )
    indices = [i for i in range(len(t_prompts))]
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
                bi = batch_input_ids[j:j+micro_train_batch_size]
                bl = batch_labels[j:j+micro_train_batch_size]
                out = get_response_log_probs(model, bi, bl, False)
                old_log_probs.append(out["log_probs"].detach())
            old_log_probs = torch.cat(old_log_probs, dim=0)

        for epoch in range(epochs_per_rollout_batch):
            optimizer.zero_grad()

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

                scaled_loss, _ = grpo_microbatch_train_step(
                    log_probs,
                    bm,
                    gradient_accumulation_steps,
                    loss_type,
                    micro_rewards,
                    micro_advantages,
                    micro_old_log_probs,
                    0.2,
                )

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Basic logging
        if i % 10 == 0:
            avg_reward = raw_rewards.mean().item()
            print(f"Step {i}: avg_reward={avg_reward:.3f}")


if __name__ == "__main__":
    train()
