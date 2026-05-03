import torch
import torch.nn as nn
from functools import partial

def relu_filter(name: str, module: torch.nn.Module) -> bool:
    """Return True if the module is a ReLU."""
    return isinstance(module, nn.ReLU)

class InputHook:
    def __init__(self, model: nn.Module, filter=relu_filter, beta=1.0):
        """
        Hook to capture input during forward pass and store pairwise distance matrices.
        Args:
            model: neural network
            filter: function to select layers (name, module) -> bool
            beta: scaling coefficient for soft VQ
        """
        self.model = model
        self.filter = filter
        self.beta = beta
        self.vq_kernel = None
        self.handles = {}
        self._register()

    def _register(self):
        for name, module in self.model.named_modules():
            if self.filter(name, module):
                hook = partial(self._hook_fn, name)
                self.handles[name] = module.register_forward_hook(hook)

    def _soft_vq(self, x):
        if self.beta==1.0:
            return (torch.sign(x)+1)/2
        else:
            return torch.sigmoid(self.beta * x / (1 - self.beta))

    def _hook_fn(self, name, module, input, output):
        x = input[0].detach()
        x = x.reshape(x.size(0), -1)
        soft_vq_x = self._soft_vq(x)
        dist_matrix = torch.cdist(soft_vq_x.double(), soft_vq_x.double(), p=1)
        if self.vq_kernel is None:
            self.vq_kernel = dist_matrix
        else:
            self.vq_kernel = self.vq_kernel + dist_matrix

    def remove(self):
        for h in self.handles.values():
            h.remove()
        self.handles.clear()

    def dump(self):
        """Return the stored distance matrices"""
        return self.dist_matrices

    def clear(self):
        self.vq_kernel = None

    def __call__(self, x):
        self.vq_kernel = None
        return self.model(x)


def kernel_alignment(Ks, Kt, eps=1e-8):
    return (Ks * Kt).sum() / (Ks.norm(p='fro') * Kt.norm(p='fro') + eps)
