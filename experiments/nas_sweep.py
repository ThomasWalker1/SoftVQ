"""E3b: zero-cost NAS metrics from the VQ kernel on NAS-Bench-201, with MeCo as a baseline.

Architectures are scored in a fixed random order and appended to a JSONL file, so a partial run
is a uniform random sample of the search space (bootstrap CIs remain valid). Sharding:
--shard i --nshards k processes every k-th architecture of that order.

Metrics of the total VQ kernel K (sum over ReLU layers, batch of 128 training images,
randomly initialized network in train mode) for each (beta, normalized):
  er    effective rank of K (Roy & Vetterli, 2007)
  logdet  log|N_A - K| (NASWOT, Mellor et al., 2021, generalized to SoftVQ; N_A = #neurons)
  frob  mean squared off-diagonal entry of K
  cond  condition number of K
MeCo (Jiang et al., 2023), our reimplementation: sum over ReLU outputs of the minimum
eigenvalue of the channel-wise Pearson correlation matrix of a single input's feature map.

    python nas_sweep.py --dataset cifar100 --shard 0 --nshards 4
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np
import torch
import torch.nn as nn

NAS_LIB = os.path.expanduser('~/terminator_splinetheory/scripts/vqd_nas')
sys.path.insert(0, NAS_LIB)
from lib.models.cell_infers.tiny_network import TinyNetwork  # noqa: E402
from lib.models.cell_searchs.genotypes import Structure  # noqa: E402

from softvq import codes, l1_kernel  # noqa: E402

BETAS = (0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0)
CONFIGS = [(b, False) for b in BETAS] + [(b, True) for b in (0.6, 0.8, 0.95)]
API_PATH = '/mnt/richb/tw78/outputs/vqd_nas/NAS-Bench-201-v1_0-e61699.pth'
DATA = os.path.expanduser('~/softvq_data')
NB201_NAME = {'cifar10': 'cifar10', 'cifar100': 'cifar100', 'ImageNet16-120': 'ImageNet16-120'}
NUM_CLASSES = {'cifar10': 10, 'cifar100': 100, 'ImageNet16-120': 120}


def probe_batch(dataset, n=128):
    if dataset == 'ImageNet16-120':
        with open('/mnt/richb/tw78/data/ImageNet16/train_data_batch_1', 'rb') as f:
            d = pickle.load(f, encoding='latin1')
        keep = np.array(d['labels']) <= 120
        x = d['data'][keep][:n].reshape(-1, 3, 16, 16)
        mean, std = np.array([122.68, 116.66, 104.01]) / 255, np.array([63.22, 61.26, 65.09]) / 255
    else:
        sub = 'cifar-10-batches-py/data_batch_1' if dataset == 'cifar10' else 'cifar-100-python/train'
        with open(os.path.join(DATA, sub), 'rb') as f:
            d = pickle.load(f, encoding='bytes')
        x = d[b'data'][:n].reshape(-1, 3, 32, 32)
        mean, std = ((np.array([0.4914, 0.4822, 0.4465]), np.array([0.2470, 0.2435, 0.2616])) if dataset == 'cifar10'
                     else (np.array([0.5071, 0.4865, 0.4409]), np.array([0.2673, 0.2564, 0.2762])))
    x = torch.tensor(x).float() / 255
    return ((x - torch.tensor(mean).view(1, 3, 1, 1).float()) / torch.tensor(std).view(1, 3, 1, 1).float()).cuda()


def kernel_metrics(K, n_neurons):
    S = torch.linalg.svdvals(K)
    p = S / S.sum()
    er = torch.exp(-(p * torch.log(p + 1e-30)).sum()).item()
    sign, logdet = torch.linalg.slogdet(n_neurons - K)
    off = K[~torch.eye(len(K), dtype=bool, device=K.device)]
    return {'er': er, 'logdet': logdet.item() if sign > 0 else float('nan'),
            'frob': off.pow(2).mean().item(), 'cond': (S[0] / S[-1].clamp_min(1e-12)).item()}


@torch.no_grad()
def score(net, x):
    Ks = {c: 0 for c in CONFIGS}
    state = {'neurons': 0, 'meco': 0.0}

    def hook(m, inp, out):
        z = inp[0].flatten(1).float()
        state['neurons'] += z.shape[1]
        for (b, nrm) in CONFIGS:
            Ks[(b, nrm)] = Ks[(b, nrm)] + l1_kernel(codes(z, b, nrm), chunk=16384)
        f = out[0].flatten(1).double()  # single input, (C, HW)
        if f.shape[1] > 1:
            C = torch.corrcoef(f)
            C = torch.nan_to_num(C, nan=0.0)
            state['meco'] += torch.linalg.eigvalsh(C)[0].item()

    hs = [m.register_forward_hook(hook) for m in net.modules() if isinstance(m, nn.ReLU)]
    net.train()
    net(x)
    for h in hs:
        h.remove()
    out = {'meco': state['meco'], 'n_neurons': state['neurons']}
    if state['neurons'] == 0:
        return out
    for (b, nrm), K in Ks.items():
        tag = f'b{b}' + ('_norm' if nrm else '')
        for k, v in kernel_metrics(K, state['neurons']).items():
            out[f'{k}_{tag}'] = v
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='cifar100', choices=list(NB201_NAME))
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--nshards', type=int, default=1)
    p.add_argument('--out', default=os.path.expanduser('~/softvq_runs/nas'))
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fn = os.path.join(a.out, f'{a.dataset}_shard{a.shard}of{a.nshards}.jsonl')
    done = set()
    if os.path.exists(fn):
        done = {json.loads(l)['index'] for l in open(fn)}

    meta_fn = os.path.join(a.out, 'nb201_meta.json')
    if os.path.exists(meta_fn):
        meta = json.load(open(meta_fn))
    else:
        from lib.nas_201_api import NASBench201API
        api = NASBench201API(API_PATH, verbose=False)
        meta = {'arch': [api.arch(i) for i in range(len(api))]}
        for ds in NB201_NAME:
            split = 'ori-test' if ds == 'cifar10' else 'x-test'
            meta[ds] = [api.arch2infos_full[i].get_metrics(NB201_NAME[ds], split, iepoch=None, is_random=False)['accuracy']
                        for i in range(len(api))]
        json.dump(meta, open(meta_fn + '.tmp', 'w'))
        os.replace(meta_fn + '.tmp', meta_fn)

    order = np.random.default_rng(0).permutation(len(meta['arch']))[a.shard::a.nshards]
    x = probe_batch(a.dataset)
    with open(fn, 'a') as f:
        for idx in order:
            idx = int(idx)
            if idx in done:
                continue
            torch.manual_seed(idx)
            net = TinyNetwork(16, 5, Structure.str2structure(meta['arch'][idx]), NUM_CLASSES[a.dataset]).cuda()
            row = {'index': idx, 'acc': meta[a.dataset][idx], **score(net, x)}
            f.write(json.dumps(row) + '\n')
            f.flush()


if __name__ == '__main__':
    main()
