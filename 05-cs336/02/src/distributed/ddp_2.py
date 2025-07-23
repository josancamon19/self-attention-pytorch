import torch
import torch.distributed as dist


class DDPWrapper(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module
        self.async_handles = []
        self._broadcast_parameters()
        self._register_gradient_hooks()

    def _broadcast_parameters(self):
        """Broadcast parameters from rank 0 to all other ranks."""
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)

    def _register_gradient_hooks(self):
        for param in self.module.parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(self._gradient_hook)

    def _gradient_hook(self, param: torch.nn.Parameter):
        if param.grad is not None:
            handle = dist.all_reduce(param.grad, op=dist.ReduceOp.AVG, async_op=True)
            self.async_handles.append(handle)

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        for handle in self.async_handles:
            handle.wait()
        self.async_handles.clear()
