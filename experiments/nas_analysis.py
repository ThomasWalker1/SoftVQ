"""E3b analysis: Spearman correlation (with 95% bootstrap CIs) between zero-cost metrics and
trained NAS-Bench-201 accuracy, for each VQ-kernel metric and beta, and for MeCo.

Writes <paper>/data/nas_<dataset>.csv with one row per beta and columns <metric>, <metric>_lo,
<metric>_hi (raw SoftVQ), <metric>_norm... (normalized, where computed), and a MeCo row file.

    python nas_analysis.py --dataset cifar100
"""
import argparse
import glob
import json
import os

import numpy as np
from scipy.stats import rankdata

BETAS = (0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0)
METRICS = ('er', 'logdet', 'frob', 'cond')


def spearman(a, b):
    ra, rb = rankdata(a), rankdata(b)
    return np.corrcoef(ra, rb)[0, 1]


def spearman_ci(x, y, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(x)
    boots = [spearman(x[i], y[i]) for i in (rng.integers(0, n, n) for _ in range(n_boot))]
    return spearman(x, y), np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='cifar100')
    p.add_argument('--runs', default=os.path.expanduser('~/softvq_runs/nas'))
    p.add_argument('--paper', default=os.path.expanduser('~/SoftVQ-paper/data'))
    p.add_argument('--n_boot', type=int, default=1000)
    a = p.parse_args()
    rows = [json.loads(l) for f in glob.glob(os.path.join(a.runs, f'{a.dataset}_shard*of8.jsonl')) for l in open(f)]
    # degenerate architectures (no ReLU neurons, e.g. only skip/pool/none) carry no VQ metrics
    ok = [r for r in rows if 'er_b1.0' in r]
    print(f'{a.dataset}: {len(rows)} architectures scored, {len(ok)} with ReLU neurons')
    acc = np.array([r['acc'] for r in ok])

    def col(key):
        v = np.array([r.get(key, np.nan) for r in ok], float)
        v[~np.isfinite(v)] = np.nanmin(v[np.isfinite(v)]) if np.isfinite(v).any() else 0
        return v

    out = os.path.join(a.paper, f'nas_{a.dataset}.csv')
    with open(out, 'w') as f:
        hdr = ['beta'] + [f'{m}{s}{c}' for m in METRICS for s in ('', '_norm') for c in ('', '_lo', '_hi')]
        f.write(','.join(hdr) + '\n')
        for b in BETAS:
            vals = [str(b)]
            for m in METRICS:
                for s in ('', '_norm'):
                    key = f'{m}_b{b}{s}'
                    if not any(key in r for r in ok):
                        vals += ['nan'] * 3
                        continue
                    # condition number: lower is better for trainability, as in prior work
                    x = -col(key) if m == 'cond' else col(key)
                    rho, lo, hi = spearman_ci(x, acc, a.n_boot)
                    vals += [f'{rho:.4f}', f'{lo:.4f}', f'{hi:.4f}']
            f.write(','.join(vals) + '\n')
    rho, lo, hi = spearman_ci(col('meco'), acc, a.n_boot)
    with open(os.path.join(a.paper, f'nas_{a.dataset}_meco.csv'), 'w') as f:
        f.write(f'metric,rho,lo,hi,n\nmeco,{rho:.4f},{lo:.4f},{hi:.4f},{len(ok)}\n')
    print(open(out).read())
    print(f'MeCo: {rho:.3f} [{lo:.3f}, {hi:.3f}]')


if __name__ == '__main__':
    main()
