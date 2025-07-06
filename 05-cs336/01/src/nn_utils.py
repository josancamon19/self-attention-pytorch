from collections.abc import Callable, Iterable
import math
import torch
import numpy as np


def data_loading(
    x: np.ndarray,
    batch_size: int,
    context_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    starting_indices = torch.randint(0, len(x) - context_length, (batch_size,))
    indices = starting_indices.unsqueeze(1) + torch.arange(context_length + 1)
    batch = torch.from_numpy(x[indices]).to(device, dtype=torch.long)
    return batch[:, :-1], batch[:, 1:]


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    path: str,
    iteration: int | None = None,  # adapter
):
    data = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(data, path)


def load_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, path: str):
    saved = torch.load(path)
    model.load_state_dict(saved["model"])
    optimizer.load_state_dict(saved["optimizer"])
    if "iteration" in saved:
        return saved["iteration"]


def cos_lr_schedule(lr_min, lr_max, warmup_steps, annealing_steps, step):
    # TODO: deeper understanding
    if step < warmup_steps:
        return (step / warmup_steps) * lr_max
    elif warmup_steps <= step <= annealing_steps:
        cos = math.cos(math.pi * (step - warmup_steps) / (annealing_steps - warmup_steps))
        return lr_min + ((1 + cos) / 2) * (lr_max - lr_min)
    else:  # step > annealing_steps
        return lr_min


# TODO: not optimal when compared to original ones, not at all, why?
def clip_gradients(params: Iterable, max_norm: float):
    grads = [p.grad for p in params if p.grad is not None]
    norms = [torch.linalg.vector_norm(g) for g in grads]
    total_norm = torch.linalg.vector_norm(torch.stack(norms))
    if total_norm < max_norm:
        return
    scale_factor = max_norm / (total_norm + 1e-6)
    [grad.data.mul_(scale_factor) for grad in grads]


def cross_entropy_loss(result: torch.Tensor, labels: torch.Tensor):
    # TODO: understand deeper, how log/exponentials cancel, how that fixes numerical instability
    # -- cases like 0.00000 in logs include an infinite
    max_vals = torch.max(result, dim=1, keepdim=True)[0]
    shifted_logits = result - max_vals
    log_sum_exp = torch.log(torch.sum(torch.exp(shifted_logits), dim=1, keepdim=True))
    log_probs = shifted_logits - log_sum_exp

    indices = torch.arange(result.shape[0])
    correct_log_probs = log_probs[indices, labels]
    # result = softmax(result, dim=1)
    return -torch.mean(correct_log_probs)


class SGD(torch.optim.Optimizer):
    # Problem (learning_rate_tuning):
    # 1e1, kept on 9.30
    # 1e2, .30, .28
    # TODO: something fun to test, train the whole model, and get losses curves for 2 epochs, on 5 different lr's
    def __init__(self, params, lr=1e-3):
        super().__init__(params, {"lr": lr})

    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t", 0)  # step
                grad = p.grad.data
                p.data -= lr / math.sqrt(t + 1) * grad  # update
                state["t"] = t + 1  # update step
        return loss
