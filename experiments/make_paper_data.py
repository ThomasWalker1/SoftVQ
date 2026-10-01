"""Convert experiment outputs into the CSVs read by the paper's pgfplots figures.

    python make_paper_data.py --paper ~/SoftVQ-paper/data
"""
import argparse
import glob
import json
import os

import numpy as np

RUNS = os.path.expanduser('~/softvq_runs')


def pretrained(out):
    fn = os.path.join(RUNS, 'pretrained', 'pretrained_alignment.json')
    if not os.path.exists(fn):
        return
    rows = json.load(open(fn))
    keys = ['cka_linear_penultimate', 'cka_linear_all_layers', 'vq_uncentered_b0.8', 'vq_centered_b0.8',
            'vq_centered_b0.8_norm', 'vq_centered_b0.5', 'vq_centered_b1.0']
    with open(os.path.join(out, 'pretrained_alignment.csv'), 'w') as f:
        f.write('model,' + ','.join(f'{k}_{t}' for k in keys for t in ('trained', 'random')) + '\n')
        for m in ['resnet18', 'resnet34', 'resnet50', 'resnet101']:
            r = {x['pretrained']: x for x in rows if x['model'] == m}
            if len(r) < 2:
                continue
            f.write(m.replace('resnet', 'ResNet') + ',' + ','.join(
                f'{r[t][k]:.4f}' for k in keys for t in (True, False)) + '\n')


def specificity(out):
    for fn in glob.glob(os.path.join(RUNS, 'sim_results', 'specificity_*.json')):
        arch = os.path.basename(fn)[len('specificity_'):-5]
        res = json.load(open(fn))
        with open(os.path.join(out, f'specificity_{arch}.csv'), 'w') as f:
            f.write('measure,acc_mean,acc_std,n_pairs\n')
            for k, v in res.items():
                a = np.array(v['accuracy_all'])
                f.write(f'{k},{a.mean():.4f},{a.std(ddof=1):.4f},{len(a)}\n')
        for k in ('cka_linear', 'vq_centered_b0.8', 'vq_centered_b0.8_norm'):
            if k in res:
                M = np.array(res[k]['mean_sim_matrix'])
                with open(os.path.join(out, f'simmatrix_{arch}_{k}.dat'), 'w') as f:
                    for i in range(M.shape[0]):
                        for j in range(M.shape[1]):
                            f.write(f'{i} {j} {M[i, j]:.4f}\n')
                        f.write('\n')


def dynamics(out):
    for fn in glob.glob(os.path.join(RUNS, 'sim_results', 'dynamics_*.json')):
        arch = os.path.basename(fn)[len('dynamics_'):-5]
        res = json.load(open(fn))
        rows = res['per_step']
        keys = [k for k in rows[0] if k not in ('step', 'rel_dev_b0.8_vs_hard')]
        with open(os.path.join(out, f'dynamics_{arch}.csv'), 'w') as f:
            f.write('step,' + ','.join(f'{k},{k}_std' for k in keys) + ',rel_dev,rel_dev_std\n')
            for r in rows:
                vals = []
                for k in keys:
                    a = np.array(r[k]['all'])
                    vals += [f'{a.mean():.4f}', f'{a.std(ddof=1):.4f}']
                d = np.array(r['rel_dev_b0.8_vs_hard'])
                f.write(f"{max(r['step'], 1)}," + ','.join(vals) + f',{d.mean():.4f},{d.std(ddof=1):.4f}\n')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--paper', default=os.path.expanduser('~/SoftVQ-paper/data'))
    a = p.parse_args()
    pretrained(a.paper)
    specificity(a.paper)
    dynamics(a.paper)


if __name__ == '__main__':
    main()
