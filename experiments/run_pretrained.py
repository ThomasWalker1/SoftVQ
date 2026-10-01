"""E1b: VQ alignment between torchvision ResNets and a pretrained ResNet152 (redo of Fig. 12b).

Compares uncentered (old) and centered (Eq. 4.1) VQ alignment, raw and RMS-normalized SoftVQ,
against linear CKA (penultimate features, and summed over all ReLU layers), for pretrained and
randomly initialized ResNet18/34/50/101. Kernels are accumulated inside the forward hooks so
that no preactivations need to be stored.

    python run_pretrained.py --out ~/softvq_runs/pretrained
"""
import argparse
import glob
import json
import os

import torch
import torch.nn as nn
import torchvision
from PIL import Image

from softvq import center, codes, cosine, disable_inplace_relu, l1_kernel, vq_alignment

CONFIGS = [(b, nrm) for b in (0.5, 0.8, 0.9, 1.0) for nrm in (False, True)]


class KernelAccumulator:
    def __init__(self, model):
        disable_inplace_relu(model)
        self.K = {c: 0 for c in CONFIGS}
        self.G_all = 0
        self.handles = [m.register_forward_hook(self._hook) for m in model.modules() if isinstance(m, nn.ReLU)]

    def _hook(self, module, inputs, output):
        z = inputs[0].detach().flatten(1).float()
        for (b, nrm) in CONFIGS:
            if b == 1.0 and nrm:  # HardVQ is already rescaling-invariant
                continue
            self.K[(b, nrm)] = self.K[(b, nrm)] + l1_kernel(codes(z, b, nrm), chunk=4096)
        h = output.detach().flatten(1).double()
        self.G_all = self.G_all + h @ h.T

    def remove(self):
        for h in self.handles:
            h.remove()


def load_probe(folder):
    tf = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256), torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    files = sorted(glob.glob(os.path.join(folder, '*.JPEG')))
    return torch.stack([tf(Image.open(f).convert('RGB')) for f in files])


@torch.no_grad()
def kernels(model, x):
    model.eval().cuda()
    acc = KernelAccumulator(model)
    feats = {}
    hook = model.avgpool.register_forward_hook(lambda m, i, o: feats.__setitem__('pen', o.flatten(1).double()))
    model(x.cuda())
    acc.remove()
    hook.remove()
    acc.K[(1.0, True)] = acc.K[(1.0, False)]
    model.cpu()
    return acc.K, acc.G_all, feats['pen'] @ feats['pen'].T


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--probe', default=os.path.expanduser('~/softvq_data/imagenet_probe'))
    p.add_argument('--out', default=os.path.expanduser('~/softvq_runs/pretrained'))
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    x = load_probe(a.probe)
    print('probe', x.shape, flush=True)

    def build(name, pretrained, seed=0):
        torch.manual_seed(seed)
        return getattr(torchvision.models, name)(weights='IMAGENET1K_V1' if pretrained else None)

    ref_K, ref_Gall, ref_Gpen = kernels(build('resnet152', True), x)
    rows = []
    for name in ['resnet18', 'resnet34', 'resnet50', 'resnet101']:
        for pretrained in (True, False):
            K, Gall, Gpen = kernels(build(name, pretrained), x)
            row = {'model': name, 'pretrained': pretrained,
                   'cka_linear_penultimate': cosine(center(Gpen), center(ref_Gpen)),
                   'cka_linear_all_layers': cosine(center(Gall), center(ref_Gall))}
            for (b, nrm) in CONFIGS:
                tag = f'b{b}' + ('_norm' if nrm else '')
                row[f'vq_uncentered_{tag}'] = vq_alignment(K[(b, nrm)], ref_K[(b, nrm)], centered=False)
                row[f'vq_centered_{tag}'] = vq_alignment(K[(b, nrm)], ref_K[(b, nrm)], centered=True)
            rows.append(row)
            print(json.dumps(row), flush=True)
            with open(os.path.join(a.out, 'pretrained_alignment.json'), 'w') as f:
                json.dump(rows, f, indent=1)


if __name__ == '__main__':
    main()
