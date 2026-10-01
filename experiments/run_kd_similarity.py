"""E5: does student-teacher similarity predict the robustness transferred by distillation?

For every distilled Imagenette student (ResNet18, SiLU) we measure, on the same checkpoint,
  - test accuracy and l_inf robust accuracy (APGD, eps=4/255, as in vqk_distill/train.py), and
  - its similarity to the teacher under each measure: layer-matched linear CKA, and VQ alignment
    (beta=1/2 is the SiLU gate; beta=1 is the step function of the preactivation), centered and
    uncentered, raw and normalized, both layer-matched and for the whole-network kernel.

    python run_kd_similarity.py --runs ~/softvq_runs/kd --ckpt student_best
"""
import argparse
import glob
import json
import os
import sys

import torch
import torch.nn as nn

KD = os.path.expanduser('~/terminator_splinetheory/scripts/vqk_distill')
sys.path.insert(0, KD)
from dataset.imagenette import get_imagenette_test_transform  # noqa: E402
from engine.utils import evaluate_robustness_inf  # noqa: E402
from models.imagenette.resnet import ResNet18  # noqa: E402
from torchvision import datasets  # noqa: E402

from softvq import center, collect, cosine, gram_linear, vq_kernels  # noqa: E402

CONFIGS = [(0.5, False), (0.5, True), (0.8, False), (1.0, False)]


def load(path):
    m = ResNet18(num_classes=10, pool=True, activation_fn=nn.SiLU)
    m.load_state_dict(torch.load(path, map_location='cpu', weights_only=False)['model'])
    return m.cuda().eval()


@torch.no_grad()
def features(model, x):
    model.with_features = True
    pre, post = collect(model, x, batch_size=100, acts=(nn.SiLU,))
    out = {'cka': [center(gram_linear(h)) for h in post]}
    for b, nrm in CONFIGS:
        out[(b, nrm)] = vq_kernels(pre, b, nrm)
    return out


def similarities(S, T):
    r = {'cka_linear_layer': sum(cosine(a, b) for a, b in zip(S['cka'], T['cka'])) / len(S['cka'])}
    for b, nrm in CONFIGS:
        tag = f'b{b}' + ('_norm' if nrm else '')
        Ks, Kt = S[(b, nrm)], T[(b, nrm)]
        r[f'vq_centered_{tag}_layer'] = sum(cosine(center(a), center(c)) for a, c in zip(Ks, Kt)) / len(Ks)
        r[f'vq_uncentered_{tag}_layer'] = sum(cosine(a, c) for a, c in zip(Ks, Kt)) / len(Ks)
        r[f'vq_centered_{tag}_net'] = cosine(center(sum(Ks)), center(sum(Kt)))
        r[f'vq_uncentered_{tag}_net'] = cosine(sum(Ks), sum(Kt))
    return r


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs', default=os.path.expanduser('~/softvq_runs/kd'))
    p.add_argument('--ckpt', default='student_best')
    p.add_argument('--n', type=int, default=500)
    p.add_argument('--out', default=os.path.expanduser('~/softvq_runs/kd_similarity.json'))
    a = p.parse_args()
    data = os.environ.get('SOFTVQ_DATA', os.path.expanduser('~/softvq_data'))
    val = datasets.Imagenette(root=data, split='val', size='160px', transform=get_imagenette_test_transform())
    loader = torch.utils.data.DataLoader(val, batch_size=64, shuffle=False, num_workers=4)
    g = torch.Generator().manual_seed(0)
    idx = torch.randperm(len(val), generator=g)[:a.n]
    x = torch.stack([val[i][0] for i in idx]).cuda()

    teacher = load(os.path.join(KD, 'models/imagenette/ckpts/rn18'))
    T = features(teacher, x)
    results = json.load(open(a.out)) if os.path.exists(a.out) else {}
    for run in sorted(glob.glob(os.path.join(a.runs, '*', a.ckpt))):
        name = os.path.basename(os.path.dirname(run))
        key = f'{name}/{a.ckpt}'
        if key in results:
            continue
        student = load(run)
        row = {'run': name, 'ckpt': a.ckpt, **similarities(features(student, x), T)}
        student.with_features = False
        correct = 0
        with torch.no_grad():
            for xb, yb in loader:
                correct += (student(xb.cuda()).argmax(1) == yb.cuda()).sum().item()
        row['test_acc'] = 100 * correct / len(val)
        row['robust_acc_linf'] = evaluate_robustness_inf(student, loader, 'cuda', eps=4. / 255.)
        results[key] = row
        print(json.dumps(row), flush=True)
        json.dump(results, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
