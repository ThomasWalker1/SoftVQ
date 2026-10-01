"""SoftVQ codes, VQ kernels and representational similarity measures.

VQ kernel (Sec. 3): K_beta[i, k] = sum over ReLU neurons of |s(x_i) - s(x_k)|, with
s = sigmoid(beta / (1 - beta) * z) for preactivation z (beta = 1 gives HardVQ = step(z)).
Normalized SoftVQ (Sec. 4) divides each preactivation by its RMS over the probe inputs.
"""
import torch
import torch.nn as nn


def disable_inplace_relu(model):
    """In-place ReLUs overwrite their input before forward hooks run, so a hook would see
    rectified values instead of preactivations. Switch them off (function unchanged)."""
    for m in model.modules():
        if isinstance(m, nn.ReLU):
            m.inplace = False
    return model


class PreactRecorder:
    """Records the input to every activation-module call (default nn.ReLU), in call order."""

    def __init__(self, model, acts=(nn.ReLU,)):
        disable_inplace_relu(model)
        self.pre, self.post = [], []
        self.handles = [m.register_forward_hook(self._hook)
                        for m in model.modules() if isinstance(m, acts)]

    def _hook(self, module, inputs, output):
        self.pre.append(inputs[0].detach().flatten(1))
        self.post.append(output.detach().flatten(1))

    def clear(self):
        self.pre, self.post = [], []

    def remove(self):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def collect(model, x, batch_size=250, acts=(nn.ReLU,)):
    """Per-activation-layer preactivations and activations for probe inputs x (lists of (n, D_l) tensors)."""
    model.eval()
    rec = PreactRecorder(model, acts)
    pre, post = None, None
    for i in range(0, len(x), batch_size):
        rec.clear()
        model(x[i:i + batch_size])
        if pre is None:
            pre, post = [[p] for p in rec.pre], [[p] for p in rec.post]
        else:
            for l, (p, q) in enumerate(zip(rec.pre, rec.post)):
                pre[l].append(p)
                post[l].append(q)
    rec.remove()
    return [torch.cat(p) for p in pre], [torch.cat(q) for q in post]


def codes(z, beta, normalize=False):
    if normalize:
        z = z / (z.pow(2).mean(0, keepdim=True).sqrt() + 1e-12)
    if beta >= 1.0:
        return (z >= 0).to(z.dtype)
    return torch.sigmoid(beta / (1.0 - beta) * z)


def l1_kernel(c, chunk=8192):
    """Pairwise l1 distances between rows of c.

    Each feature chunk is computed in float32 (entries are in [0, 1], so a chunk sum is at most
    `chunk`) and accumulated in float64; FP64 cdist is ~30x slower on consumer/Turing GPUs.
    """
    n = c.shape[0]
    K = torch.zeros(n, n, dtype=torch.float64, device=c.device)
    c = c.float()
    for j in range(0, c.shape[1], chunk):
        K += torch.cdist(c[:, j:j + chunk], c[:, j:j + chunk], p=1).double()
    return K


@torch.no_grad()
def vq_kernels(pre, beta, normalize=False):
    """List of per-layer VQ kernels (n, n)."""
    return [l1_kernel(codes(z.float(), beta, normalize)) for z in pre]


def center(K):
    n = K.shape[0]
    H = torch.eye(n, dtype=K.dtype, device=K.device) - 1.0 / n
    return H @ K @ H


def cosine(A, B):
    return ((A * B).sum() / (A.norm() * B.norm() + 1e-30)).item()


def vq_alignment(Ka, Kb, centered=True):
    """Centered VQ alignment (Eq. 4.1) of two distance matrices; centered=False is the old loss."""
    if centered:
        return cosine(-0.5 * center(Ka), -0.5 * center(Kb))
    return cosine(Ka, Kb)


def gram_linear(X):
    X = X.double()
    return X @ X.T


def gram_rbf(X, frac=0.5):
    """RBF Gram with bandwidth frac * median pairwise distance (Kornblith et al., 2019)."""
    X = X.double()
    D2 = torch.cdist(X, X).pow(2)
    sigma2 = (frac * D2[D2 > 0].sqrt().median()) ** 2
    return torch.exp(-D2 / (2 * sigma2))


def cka(Ga, Gb):
    return cosine(center(Ga), center(Gb))


def colnorm(X):
    """Divide each feature (column) by its norm over the inputs: invariant to positive rescaling."""
    X = X.double()
    return X / (X.norm(dim=0, keepdim=True) + 1e-12)


def gram_colnorm_linear(X):
    """Linear Gram of column-normalized features (rescaling-invariant CKA baseline)."""
    Xn = colnorm(X)
    return Xn @ Xn.T


def gram_max(X, chunk=256):
    """Max kernel of GReLU-CKA (mu-CKA, Godfrey et al., 2022): rows centered, columns normalized,
    K_ij = max_k x_ik x_jk. Not positive semidefinite in general."""
    X = X.double()
    X = colnorm(X - X.mean(0, keepdim=True)).float()
    n = X.shape[0]
    K = torch.full((n, n), -float('inf'), device=X.device)
    for j in range(0, X.shape[1], chunk):
        c = X[:, j:j + chunk]
        K = torch.maximum(K, (c[:, None, :] * c[None, :, :]).amax(-1))
    return K.double()

