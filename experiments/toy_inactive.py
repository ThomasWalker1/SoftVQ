"""Exact illustration that post-ReLU measures cannot see the inactive side of a neuron.

One hidden ReLU layer, inputs X (n x d). For neuron j with active set A_j = {i : z_ij > 0}, any
perturbation v of (w_j, b_j) orthogonal to {[x_i, 1] : i in A_j} leaves z_ij unchanged on A_j; for
small enough step it keeps the inactive points inactive. Post-ReLU activations on X, and hence CKA,
column-normalized CKA, GReLU-CKA and HardVQ alignment, are then *exactly* unchanged, although the
hyperplane, and hence the partition, moves. Normalized SoftVQ sees the change (Thm. 3.3).

We compare such "inactive-side" perturbations with random perturbations of the same norm.

    python toy_inactive.py
"""
import json
import os

import torch

from softvq import center, cosine, gram_colnorm_linear, gram_linear, gram_max, vq_kernels

torch.set_default_dtype(torch.float64)


def all_measures(Z):
    H = torch.relu(Z)
    out = {'cka_linear': center(gram_linear(H)), 'cka_colnorm': center(gram_colnorm_linear(H)),
           'mu_cka': center(gram_max(H)),
           'vq_b1.0': -0.5 * center(vq_kernels([Z], 1.0)[0]),
           'vq_b0.8_norm': -0.5 * center(vq_kernels([Z], 0.8, True)[0])}
    return out


def inactive_direction(X1, z, g):
    """Unit direction in (w, b) orthogonal to the augmented active inputs, or None."""
    A = X1[z > 0]
    D = X1.shape[1]
    if len(A) >= D:
        return None
    # orthonormal basis of the null space of A
    if len(A):
        _, _, Vh = torch.linalg.svd(A, full_matrices=True)
        N = Vh[len(A):]
    else:
        N = torch.eye(D)
    v = N.T @ torch.randn(N.shape[0], generator=g)
    return v / v.norm()


def main(d=20, n=21, m=200, eps_list=(0.01, 0.03, 0.1, 0.3, 1.0), trials=20, out=None):
    rows = []
    for t in range(trials):
        g = torch.Generator().manual_seed(t)
        X = torch.randn(n, d, generator=g)
        X1 = torch.cat([X, torch.ones(n, 1)], 1)
        W = torch.randn(m, d, generator=g) / d ** 0.5
        b = torch.randn(m, generator=g) - 0.5        # ~30% of units active per input
        P = torch.cat([W, b[:, None]], 1)            # (m, d+1) hyperplane parameters
        Z = X1 @ P.T
        base = all_measures(Z)
        V = torch.zeros_like(P)
        for j in range(m):
            v = inactive_direction(X1, Z[:, j], g)
            V[j] = 0 if v is None else v * P[j].norm()   # relative size eps per neuron
        R = torch.randn(P.shape, generator=g)
        R = R / R.norm(dim=1, keepdim=True) * P.norm(dim=1, keepdim=True)
        for eps in eps_list:
            for kind, D in [('inactive', V), ('random', R)]:
                Pp = P + eps * D
                if kind == 'inactive':
                    # shrink per neuron until no inactive point becomes active (exactness)
                    for j in range(m):
                        s = 1.0
                        while ((X1 @ (P[j] + s * eps * D[j]) > 0) != (Z[:, j] > 0)).any():
                            s /= 2
                        Pp[j] = P[j] + s * eps * D[j]
                Zp = X1 @ Pp.T
                mp = all_measures(Zp)
                row = {'trial': t, 'eps': eps, 'kind': kind,
                       'rel_param_change': ((Pp - P).norm() / P.norm()).item(),
                       'max_post_change': (torch.relu(Zp) - torch.relu(Z)).abs().max().item(),
                       'frac_neurons_moved': ((Pp - P).norm(dim=1) > 0).float().mean().item()}
                for k in base:
                    row[k] = cosine(base[k], mp[k])
                rows.append(row)
    summary = {}
    for kind in ('inactive', 'random'):
        for eps in eps_list:
            rs = [r for r in rows if r['kind'] == kind and r['eps'] == eps]
            summary[f'{kind}_{eps}'] = {k: sum(r[k] for r in rs) / len(rs) for k in rs[0] if k not in ('trial', 'eps', 'kind')}
            print(kind, eps, {k: round(v, 6) for k, v in summary[f'{kind}_{eps}'].items()})
    if out:
        json.dump({'d': d, 'n': n, 'm': m, 'rows': rows, 'summary': summary}, open(out, 'w'))


if __name__ == '__main__':
    main(out=os.path.expanduser('~/softvq_runs/sim_results/toy_inactive.json'))
