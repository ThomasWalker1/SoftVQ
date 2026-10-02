"""Resolution of rescaling-invariant similarity measures as a function of the number of probe inputs.

SoftVQ resolves a neuron's hyperplane from any d+1 affinely independent inputs (Thm. 3.3), whereas
post-ReLU activations constrain it only through the inputs on which the neuron is active, and HardVQ
only through signs (Prop. 2.2). We therefore repeat the specificity test (Kornblith et al., 2019) with
n probe inputs, for n from 10 to 500, over several random draws of the probe set.

    python run_resolution.py --arch resnet18 --seeds 0-9
"""
import argparse
import itertools
import json
import os

import torch

from run_similarity import load_model, parse_seeds, sim_matrix
from softvq import center, collect, gram_colnorm_linear, gram_linear, gram_max, vq_kernels
from train_cifar import load_cifar10

NS = [10, 20, 50, 100, 200, 500]
DRAWS = 5


def measures_from(pre, post):
    out = {'cka_linear': [center(gram_linear(h)).cpu() for h in post],
           'cka_colnorm': [center(gram_colnorm_linear(h)).cpu() for h in post],
           'mu_cka': [center(gram_max(h)).cpu() for h in post]}
    for b, nrm in [(0.8, True), (0.8, False), (1.0, False)]:
        tag = f'b{b}' + ('_norm' if nrm else '')
        out[f'vq_centered_{tag}'] = [(-0.5 * center(K)).cpu() for K in vq_kernels(pre, b, nrm)]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--arch', default='resnet18')
    p.add_argument('--seeds', default='0-9')
    p.add_argument('--runs', default=os.path.expanduser('~/softvq_runs/sim'))
    p.add_argument('--out', default=os.path.expanduser('~/softvq_runs/sim_results'))
    a = p.parse_args()
    seeds = parse_seeds(a.seeds)
    _, (xte, _) = load_cifar10(os.path.expanduser('~/softvq_data/cifar-10-batches-py'), 'cuda')
    g = torch.Generator().manual_seed(0)
    x = xte[torch.randperm(len(xte), generator=g)[:max(NS)].cuda()]
    # probe subsets, shared by all models
    subsets = {(n, d): torch.randperm(max(NS), generator=g)[:n] for n in NS for d in range(DRAWS if n < max(NS) else 1)}

    def ckpt(s):
        d = os.path.join(a.runs, f'{a.arch}-s{s}')
        return os.path.join(d, sorted([f for f in os.listdir(d) if f.startswith('step')], key=lambda f: int(f[4:-3]))[-1])

    M = {}
    for s in seeds:
        pre, post = collect(load_model(a.arch, ckpt(s)), x)
        M[s] = {k: measures_from([z[idx.cuda()] for z in pre], [h[idx.cuda()] for h in post]) for k, idx in subsets.items()}
        del pre, post
        torch.cuda.empty_cache()
        print('collected seed', s, flush=True)

    names = list(next(iter(M[seeds[0]].values())))
    res = {'ns': NS, 'draws': DRAWS, 'per_n': []}
    for n in NS:
        row = {'n': n}
        for name in names:
            accs = []
            for (nn, d) in subsets:
                if nn != n:
                    continue
                for s1, s2 in itertools.combinations(seeds, 2):
                    S = sim_matrix(M[s1][(n, d)][name], M[s2][(n, d)][name])
                    accs.append((S.argmax(1) == torch.arange(len(S))).float().mean().item())
            row[name] = {'mean': sum(accs) / len(accs), 'all': accs}
        res['per_n'].append(row)
        print(json.dumps({k: (round(v['mean'], 3) if isinstance(v, dict) else v) for k, v in row.items()}), flush=True)
    with open(os.path.join(a.out, f'resolution_{a.arch}.json'), 'w') as f:
        json.dump(res, f)


if __name__ == '__main__':
    main()
