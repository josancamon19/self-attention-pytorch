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


def single_data_loading(
    x: np.ndarray,
    batch_size: int,
    steps: int,
    context_length: int,
    device: torch.device,
):
    starting_indices = torch.randint(0, len(x) - context_length, (steps, batch_size))
    indices = starting_indices.unsqueeze(-1) + torch.arange(context_length + 1)
    batches = torch.from_numpy(x[indices]).to(device, dtype=torch.long)
    print("single_data_loading", batches.shape)
    return batches[:, :, :-1], batches[:, :, 1:]  # inputs, labels


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    path: str,
    args: dict = {},
    iteration: int | None = None,  # adapter
):
    data = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "iteration": iteration, "args": args}
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


def inverse_sqrt_schedule(_, lr_max, warmup_steps, __, step):
    if step < warmup_steps:
        return lr_max * (step / warmup_steps)
    else:
        return lr_max * (warmup_steps / step) ** 0.5


def constant_with_warmup(_, lr_max, warmup_steps, __, step):
    """What BERT actually used"""
    if step < warmup_steps:
        return (step / warmup_steps) * lr_max
    else:
        return lr_max  # No decay!


def linear_lr_schedule(lr_min, lr_max, warmup_steps, annealing_steps, step):
    """Simple linear decay after warmup"""
    if step < warmup_steps:
        return (step / warmup_steps) * lr_max
    elif warmup_steps <= step <= annealing_steps:
        progress = (step - warmup_steps) / (annealing_steps - warmup_steps)
        return lr_max - progress * (lr_max - lr_min)
    else:
        return lr_min


def cosine_with_restarts(lr_max, lr_min, warmup_steps, _, step, restart_interval=5000):
    cycle_step = step % restart_interval
    if cycle_step < warmup_steps:
        return lr_max * (cycle_step / warmup_steps)
    else:
        cos_inner = math.pi * (cycle_step - warmup_steps) / (restart_interval - warmup_steps)
        return lr_min + (lr_max - lr_min) * 0.5 * (1 + math.cos(cos_inner))


# TODO: not optimal when compared to original ones, not at all, why?
# def clip_gradients(params: Iterable, max_norm: float):
#     grads = [p.grad for p in params if p.grad is not None]
#     norms = [torch.linalg.vector_norm(g) for g in grads]
#     # TODO: is it the stacking
#     total_norm = torch.linalg.vector_norm(torch.stack(norms))
#     # TODO: conditionals, requires gpu to sync with python program,
#     # narrowing, tensor into a conditional, max vs torch.max()
#     # if it's built in python (not torch), communication overhead + wait for python + torch runtime.
#     if total_norm < max_norm:  # this is not legal in jax
#         return  # if the data depends on the tensor it's bad
#     scale_factor = max_norm / (total_norm + 1e-6)
#     [grad.data.mul_(scale_factor) for grad in grads]
#     return total_norm


# test this at some point, thought above function was slowing down training
# - but at the end was 7:08 to 7:13 per epoch, 1.1%, meh
def clip_gradients(params: Iterable, max_norm: float):
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return torch.tensor(0.0)
    grad_norms = [torch.linalg.vector_norm(g) for g in grads]
    total_norm = torch.linalg.vector_norm(torch.stack(grad_norms))
    clip_coef = torch.clamp(max_norm / (total_norm + 1e-6), max=1.0)
    for grad in grads:
        grad.data.mul_(clip_coef)
    return total_norm


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
    # -- Marcel: There's a flash implementation of cross entropy, that is just cuda code. Uses online softmax trick.
    return -torch.mean(correct_log_probs)


def cross_entropy_loss2(result: torch.Tensor, labels: torch.Tensor):
    # better than above but still doesn't match cuda imp
    log_sum_exp = torch.logsumexp(result, dim=1)
    correct_logits = result.gather(1, labels.unsqueeze(1)).squeeze(1)
    return -torch.mean(correct_logits - log_sum_exp)


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

                # Marcel: you could have faster time per parameter, but not really done
                state = self.state[p]
                t = state.get("t", 0)  # step
                grad = p.grad.data
                p.data -= lr / math.sqrt(t + 1) * grad  # update
                state["t"] = t + 1  # update step
        return loss


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        weight_decay=0.01,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
    ):
        # lr, weight_decay stored for each param group.
        self.b1 = betas[0]
        self.b2 = betas[1]
        self.b1d = 1 - self.b1
        self.b2d = 1 - self.b2
        self.eps = eps
        self.t = 1
        super().__init__(params, {"lr": lr, "weight_decay": weight_decay})

    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            lr_t = lr * (math.sqrt(1 - self.b2**self.t) / (1 - self.b1**self.t))
            wd_factor = 1 - lr * weight_decay

            # print("AdamW.group", group.keys(), lr, weight_decay, len(group["params"]))
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                if "m" not in state:
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)

                # TODO: next steps improve performance
                #   1. Eliminate torch.sqrt(state["v"]) + self.eps temporary

                state["m"].mul_(self.b1).add_(p.grad, alpha=self.b1d)
                state["v"].mul_(self.b2)
                state["v"].addcmul_(p.grad, p.grad, value=self.b2d)

                p.data.addcdiv_(state["m"], torch.sqrt(state["v"]) + self.eps, value=-lr_t)
                p.data.mul_(wd_factor)

        self.t += 1
        return loss


# def softmax(tensor: torch.Tensor, dim: int = 0):
#     max_vals = torch.max(tensor, dim=dim, keepdim=True)[0]
#     num_part = torch.exp(tensor - max_vals)
#     div_term = torch.sum(num_part, dim=dim, keepdim=True)
#     return num_part / div_term


def softmax(tensor: torch.Tensor, dim: int = 0):
    return torch.softmax(tensor, dim=dim)
