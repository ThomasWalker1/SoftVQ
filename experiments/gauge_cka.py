"""How much does a function-preserving reparametrization change CKA between two *different* networks?

For BN-free CNNs, rescaling output channel j of conv layer l by c_j > 0 (and compensating in the
next layer) leaves the network function unchanged and multiplies the layer-l activations of
channel j by c_j, without affecting other layers. Linear CKA between layer l of network A and
layer l of network B therefore depends on the per-channel scales c of B through
    G_B(c) = sum_j c_j^2 G_{B,j},   G_{B,j} = gram of channel j's activations,
and we maximize / minimize CKA(G_A, G_B(c)) over log c in [-L, L] by gradient ascent/descent.
Normalized VQ alignment is exactly invariant to c (Prop. 4.2) and is reported for reference.
The optimized scales are finally applied to the actual network to verify that the function is
unchanged and that the CKA computed from the rescaled network matches.

    python gauge_cka.py --seeds 0-4 --bound 2.3
"""
import argparse
import itertools
import json
import os

import torch

from run_similarity import load_model, parse_seeds, rescale_plaincnn
from softvq import center, collect, cosine, gram_linear, vq_kernels
from train_cifar import load_cifar10


def channel_grams(h, channels):
    """h: (n, C*H*W) post-activations, channel-major -> (C, n, n) per-channel grams."""
    n = h.shape[0]
    hc = h.view(n, channels, -1).transpose(0, 1).double()  # (C, n, HW)
    return hc @ hc.transpose(1, 2)


def optimize_cka(GA, Gch, sign, bound, steps=300, lr=0.05):
    """sign=+1 maximizes, -1 minimizes CKA(GA, sum_j exp(2 u_j) Gch_j) over |u_j| <= bound."""
    GAc = center(GA)
    u = torch.zeros(Gch.shape[0], dtype=torch.float64, device=Gch.device, requires_grad=True)
    opt = torch.optim.Adam([u], lr=lr)

    def cka_of(u):
        GB = torch.einsum('c,cij->ij', torch.exp(2 * u), Gch)
        GBc = center(GB)
        return (GAc * GBc).sum() / (GAc.norm() * GBc.norm())

    for _ in range(steps):
        opt.zero_grad()
        loss = -sign * cka_of(u)
        loss.backward()
        opt.step()
        with torch.no_grad():
            u.clamp_(-bound, bound)
    with torch.no_grad():
        return cka_of(u).item(), u.detach()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', default='0-4')
    p.add_argument('--bound', type=float, default=2.3, help='max |log c| (2.3 ~ factor 10)')
    p.add_argument('--n', type=int, default=500)
    p.add_argument('--runs', default=os.path.expanduser('~/softvq_runs/sim'))
    p.add_argument('--out', default=os.path.expanduser('~/softvq_runs/sim_results/gauge_cka_plaincnn.json'))
    a = p.parse_args()
    seeds = parse_seeds(a.seeds)
    _, (xte, _) = load_cifar10(os.path.expanduser('~/softvq_data/cifar-10-batches-py'), 'cuda')
    g = torch.Generator().manual_seed(0)
    x = xte[torch.randperm(len(xte), generator=g)[:a.n].cuda()]

    def ckpt(s):
        d = os.path.join(a.runs, f'plaincnn-s{s}')
        return os.path.join(d, sorted(os.listdir(d), key=lambda f: int(f[4:-3]) if f.startswith('step') else -1)[-1])

    models = {s: load_model('plaincnn', ckpt(s)) for s in seeds}
    feats = {}
    for s, m in models.items():
        pre, post = collect(m, x)
        convs = [l for l in m.features if isinstance(l, torch.nn.Conv2d)]
        feats[s] = {'post': post, 'C': [c.out_channels for c in convs],
                    'vq': [-0.5 * center(K) for K in vq_kernels(pre, 0.8, normalize=True)]}

    results = []
    for sA, sB in itertools.combinations(seeds, 2):
        FA, FB = feats[sA], feats[sB]
        row = {'pair': [sA, sB], 'layers': []}
        u_max, u_min = [], []
        for l, (hA, hB) in enumerate(zip(FA['post'], FB['post'])):
            GA = gram_linear(hA)
            Gch = channel_grams(hB, FB['C'][l])
            orig = cosine(center(GA), center(Gch.sum(0)))
            hi, uh = optimize_cka(GA, Gch, +1, a.bound)
            lo, ul = optimize_cka(GA, Gch, -1, a.bound)
            u_max.append(uh)
            u_min.append(ul)
            row['layers'].append({'layer': l, 'cka_orig': orig, 'cka_max': hi, 'cka_min': lo,
                                  'vq_norm_b0.8': cosine(FA['vq'][l], FB['vq'][l])})
        # verify on the real network: apply the minimizing scales to network B
        mB = rescale_with(models[sB], u_min)
        with torch.no_grad():
            row['max_logit_diff_min'] = (models[sB](x[:200]) - mB(x[:200])).abs().max().item()
        _, postR = collect(mB, x)
        row['verify_cka_min'] = [cosine(center(gram_linear(hA)), center(gram_linear(hR)))
                                 for hA, hR in zip(FA['post'], postR)]
        results.append(row)
        print(json.dumps({'pair': row['pair'], 'logit_diff': row['max_logit_diff_min'],
                          'per_layer': [(round(r['cka_min'], 3), round(r['cka_orig'], 3), round(r['cka_max'], 3),
                                         round(r['vq_norm_b0.8'], 3)) for r in row['layers']],
                          'verify_min': [round(v, 3) for v in row['verify_cka_min']]}), flush=True)
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump({'bound': a.bound, 'results': results}, open(a.out, 'w'))


def rescale_with(model, us):
    """Apply per-layer log-scales us[l] (one per output channel of conv l) function-preservingly."""
    import copy
    m = copy.deepcopy(model)
    convs = [l for l in m.features if isinstance(l, torch.nn.Conv2d)]
    with torch.no_grad():
        for i, (conv, u) in enumerate(zip(convs, us)):
            c = torch.exp(u).float().to(conv.weight.device)
            conv.weight.mul_(c.view(-1, 1, 1, 1))
            conv.bias.mul_(c)
            if i + 1 < len(convs):
                convs[i + 1].weight.div_(c.view(1, -1, 1, 1))
            else:
                m.fc.weight.div_(c.repeat_interleave(m.fc.in_features // conv.out_channels).view(1, -1))
    return m.eval()


if __name__ == '__main__':
    main()
