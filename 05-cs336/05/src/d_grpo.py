import pdb
from typing import Literal
import torch

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
    rewards = torch.tensor(rewards).view(-1, group_size)
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
    return -raw_rewards_or_advantages * policy_log_probs


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
