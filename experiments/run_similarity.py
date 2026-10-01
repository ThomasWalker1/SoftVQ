"""E1a (specificity test) and E2 (cross-seed convergence during training) on CIFAR10 models.

Specificity (Kornblith et al., 2019): for two independently trained networks, a good similarity
measure should match layer i of one network to layer i of the other. We report the fraction of
layers whose most similar layer in the other network is the corresponding layer, averaged over
pairs of seeds.

Dynamics: layer-matched similarity between seeds, averaged over layers, at each checkpoint.

    python run_similarity.py --arch resnet18 --seeds 0-9 --mode specificity
    python run_similarity.py --arch resnet18 --seeds 0-4 --mode dynamics
"""
import argparse
import glob
import itertools
import json
import os
import re

import torch

from softvq import center, collect, cosine, gram_linear, gram_rbf, vq_kernels
from train_cifar import ARCHS, load_cifar10

SPEC_CONFIGS = [(b, n) for b in (0.5, 0.8, 0.9, 0.95, 1.0) for n in (False, True) if not (b == 1.0 and n)]
DYN_CONFIGS = [(0.8, False), (0.8, True), (1.0, False)]


def measures(model, x, configs):
    """Per-layer centered Gram-type matrices for every measure (all moved to CPU, float64)."""
    pre, post = collect(model, x)
    out = {'cka_linear': [center(gram_linear(h)).cpu() for h in post],
           'cka_rbf': [center(gram_rbf(h)).cpu() for h in post]}
    for b, nrm in configs:
        Ks = vq_kernels(pre, b, nrm)
        tag = f'b{b}' + ('_norm' if nrm else '')
        out[f'vq_centered_{tag}'] = [(-0.5 * center(K)).cpu() for K in Ks]
        if b == 0.8 and not nrm:  # uncentered alignment as in the original KD loss / Fig. 12b
            out['vq_uncentered_b0.8'] = [K.cpu() for K in Ks]
        if b == 1.0:
            out['_K1'] = [K.cpu() for K in Ks]
        if b == 0.8 and not nrm:
            out['_K08'] = [K.cpu() for K in Ks]
    return out


def rescale_plaincnn(model, sigma, seed):
    """Function-preserving positive rescaling of every hidden channel of a PlainCNN:
    conv_l output channel j scaled by c_j ~ logNormal(0, sigma), consumer weights by 1/c_j."""
    import copy
    m = copy.deepcopy(model)
    g = torch.Generator().manual_seed(seed)
    convs = [l for l in m.features if isinstance(l, torch.nn.Conv2d)]
    with torch.no_grad():
        for i, conv in enumerate(convs):
            c = torch.exp(sigma * torch.randn(conv.out_channels, generator=g)).to(conv.weight.device)
            conv.weight.mul_(c.view(-1, 1, 1, 1))
            conv.bias.mul_(c)
            if i + 1 < len(convs):
                convs[i + 1].weight.div_(c.view(1, -1, 1, 1))
            else:  # flatten is channel-major
                m.fc.weight.div_(c.repeat_interleave(m.fc.in_features // conv.out_channels).view(1, -1))
    return m


def sim_matrix(A, B):
    return torch.tensor([[cosine(a, b) for b in B] for a in A])


def load_model(arch, path):
    m = ARCHS[arch]().cuda()
    m.load_state_dict(torch.load(path, map_location='cuda'))
    return m


def parse_seeds(s):
    """'0-9' or '0,4,7'."""
    if ',' in s:
        return [int(v) for v in s.split(',')]
    a, b = s.split('-') if '-' in s else (s, s)
    return list(range(int(a), int(b) + 1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--arch', default='resnet18')
    p.add_argument('--seeds', default='0-9')
    p.add_argument('--mode', choices=['specificity', 'dynamics', 'invariance'], default='specificity')
    p.add_argument('--n', type=int, default=500)
    p.add_argument('--runs', default=os.path.expanduser('~/softvq_runs/sim'))
    p.add_argument('--out', default=os.path.expanduser('~/softvq_runs/sim_results'))
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    seeds = parse_seeds(a.seeds)
    _, (xte, _) = load_cifar10(os.path.expanduser('~/softvq_data/cifar-10-batches-py'), 'cuda')
    g = torch.Generator().manual_seed(0)
    x = xte[torch.randperm(len(xte), generator=g)[:a.n].cuda()]

    def ckpts(seed):
        fs = glob.glob(os.path.join(a.runs, f'{a.arch}-s{seed}', 'step*.pt'))
        return sorted(fs, key=lambda f: int(re.search(r'step(\d+)', f).group(1)))

    if a.mode == 'specificity':
        M = {s: measures(load_model(a.arch, ckpts(s)[-1]), x, SPEC_CONFIGS) for s in seeds}
        names = [k for k in M[seeds[0]] if not k.startswith('_')]
        res = {}
        for name in names:
            accs, mats = [], []
            for s1, s2 in itertools.combinations(seeds, 2):
                S = sim_matrix(M[s1][name], M[s2][name])
                accs.append((S.argmax(1) == torch.arange(len(S))).float().mean().item())
                mats.append(S)
            res[name] = {'accuracy_mean': sum(accs) / len(accs), 'accuracy_all': accs,
                         'mean_sim_matrix': torch.stack(mats).mean(0).tolist()}
            print(f'{name:28s} specificity {res[name]["accuracy_mean"]:.3f}', flush=True)
        fn = f'specificity_{a.arch}.json'
    elif a.mode == 'invariance':
        assert a.arch == 'plaincnn'
        res = {'sigmas': [0.0, 0.25, 0.5, 1.0, 2.0], 'per_sigma': []}
        for sigma in res['sigmas']:
            row = {'sigma': sigma}
            for s in seeds:
                base = load_model(a.arch, ckpts(s)[-1])
                resc = rescale_plaincnn(base, sigma, seed=1000 + s)
                with torch.no_grad():
                    diff = (base.eval()(x[:200]) - resc.eval()(x[:200])).abs().max().item()
                Mb, Mr = measures(base, x, SPEC_CONFIGS), measures(resc, x, SPEC_CONFIGS)
                for name in [k for k in Mb if not k.startswith('_')]:
                    v = sum(cosine(u, w) for u, w in zip(Mb[name], Mr[name])) / len(Mb[name])
                    row.setdefault(name, []).append(v)
                row.setdefault('max_logit_diff', []).append(diff)
            res['per_sigma'].append(row)
            print(json.dumps({k: (sum(v) / len(v) if isinstance(v, list) else v) for k, v in row.items()}), flush=True)
        fn = f'invariance_{a.arch}.json'
    else:
        steps = [int(re.search(r'step(\d+)', f).group(1)) for f in ckpts(seeds[0])]
        res = {'steps': steps, 'per_step': []}
        for i, step in enumerate(steps):
            M = {s: measures(load_model(a.arch, ckpts(s)[i]), x, DYN_CONFIGS) for s in seeds}
            names = [k for k in M[seeds[0]] if not k.startswith('_')]
            row = {'step': step}
            for name in names:
                vals = [sum(cosine(u, v) for u, v in zip(M[s1][name], M[s2][name])) / len(M[s1][name])
                        for s1, s2 in itertools.combinations(seeds, 2)]
                row[name] = {'mean': sum(vals) / len(vals), 'all': vals}
            # soft-to-hard kernel deviation within each network (Fig. 3), relative Frobenius
            row['rel_dev_b0.8_vs_hard'] = [
                (sum((k8 - k1).norm() ** 2 for k8, k1 in zip(M[s]['_K08'], M[s]['_K1'])).sqrt()
                 / sum(k1.norm() ** 2 for k1 in M[s]['_K1']).sqrt()).item() for s in seeds]
            res['per_step'].append(row)
            print(json.dumps({k: (v['mean'] if isinstance(v, dict) else v) for k, v in row.items()}), flush=True)
        fn = f'dynamics_{a.arch}.json'
    with open(os.path.join(a.out, fn), 'w') as f:
        json.dump(res, f)


if __name__ == '__main__':
    main()
