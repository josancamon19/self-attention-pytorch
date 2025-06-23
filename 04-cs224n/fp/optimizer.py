from typing import Callable, Iterable, Tuple
import math

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                "Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0])
            )
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                "Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1])
            )
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            correct_bias=correct_bias,
        )
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        "Adam does not support sparse gradients, please consider SparseAdam instead"
                    )
                # print("step p:", p)
                # print("step grad:", grad)

                # State should be stored in this dictionary.
                state = self.state[p]
                # print("step state:", state)

                # Access hyperparameters from the `group` dictionary.
                lr, betas, eps, weight_decay = (
                    group["lr"],
                    group["betas"],
                    group["eps"],
                    group["weight_decay"],
                )
                # print("step hyperparams:", alpha, betas, eps, weight_decay)
                b1, b2 = betas

                if len(state) == 0:
                    state["step"] = 0
                    state["mt"] = torch.zeros_like(grad)  # First moment (m)
                    state["vt"] = torch.zeros_like(grad)  # Second moment (v)
                
                state["step"] += 1
                # Update moments in-place
                state["mt"].mul_(b1).add_(grad, alpha=1 - b1)
                state["vt"].mul_(b2).addcmul_(grad, grad, value=1 - b2)

                # Compute bias correction factor
                bias_correction = math.sqrt(1 - b2 ** state["step"]) / (1 - b1 ** state["step"])

                # Apply main update in-place
                denom = (state["vt"].sqrt() / math.sqrt(1 - b2 ** state["step"])).add_(eps)
                step_size = lr * bias_correction
                p.data.addcdiv_(state["mt"], denom, value=-step_size)

                # Apply weight decay in-place
                if weight_decay != 0:
                    p.data.mul_(1 - weight_decay * lr)

                ### TODO: Complete the implementation of AdamW here, reading and saving
                ###       your state in the `state` dictionary above.
                ###       The hyperparameters can be read from the `group` dictionary
                ###       (they are lr, betas, eps, weight_decay, as saved in the constructor).
                ###
                ###       To complete this implementation:
                ###       1. Update the first and second moments of the gradients.
                ###       2. Apply bias correction
                ###          (using the "efficient version" given in https://arxiv.org/abs/1412.6980;
                ###          also given in the pseudo-code in the project description).
                ###       3. Update parameters (p.data).
                ###       4. Apply weight decay after the main gradient-based updates.
                ###
                ###       Refer to the default project handout for more details.
                ### YOUR CODE HERE
                # raise NotImplementedError

        return loss
