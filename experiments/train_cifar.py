"""Train CIFAR10 models for the similarity experiments (E1a specificity, E2 dynamics).

Data are held on the GPU with random-crop + flip augmentation done on the GPU, so no
dataloader workers are needed. Checkpoints are saved at log-spaced steps.

    python train_cifar.py --arch resnet18 --seed 0 --out ~/softvq_runs/sim
"""
import argparse
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)


def load_cifar10(root, device):
    def load(files):
        xs, ys = [], []
        for f in files:
            with open(os.path.join(root, f), 'rb') as fh:
                d = pickle.load(fh, encoding='bytes')
            xs.append(d[b'data'])
            ys += d[b'labels']
        x = torch.tensor(np.concatenate(xs)).view(-1, 3, 32, 32).float() / 255
        return ((x - MEAN) / STD).to(device), torch.tensor(ys).to(device)
    return load([f'data_batch_{i}' for i in range(1, 6)]), load(['test_batch'])


class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.relu2 = nn.ReLU()
        self.short = nn.Sequential()
        if stride != 1 or cin != cout:
            self.short = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        return self.relu2(self.bn2(self.conv2(out)) + self.short(x))


class ResNet18(nn.Module):
    """CIFAR ResNet18 with a separate nn.ReLU per activation (17 ReLU layers)."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU())
        layers, cin = [], 64
        for cout, stride in [(64, 1), (64, 1), (128, 2), (128, 1), (256, 2), (256, 1), (512, 2), (512, 1)]:
            layers.append(BasicBlock(cin, cout, stride))
            cin = cout
        self.layers = nn.Sequential(*layers)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.layers(self.stem(x))
        return self.fc(F.adaptive_avg_pool2d(x, 1).flatten(1))


class PlainCNN(nn.Module):
    """VGG-style CNN without normalization (8 ReLU layers): rescaling symmetry is not absorbed by BN."""

    def __init__(self, num_classes=10, width=64):
        super().__init__()
        cfg = [width, width, 'M', 2 * width, 2 * width, 'M', 4 * width, 4 * width, 'M', 8 * width, 8 * width, 'M']
        layers, cin = [], 3
        for c in cfg:
            if c == 'M':
                layers.append(nn.MaxPool2d(2))
            else:
                layers += [nn.Conv2d(cin, c, 3, 1, 1), nn.ReLU()]
                cin = c
        self.features = nn.Sequential(*layers)
        self.fc = nn.Linear(cin * 4, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.fc(self.features(x).flatten(1))


ARCHS = {'resnet18': ResNet18, 'plaincnn': PlainCNN}


def augment(x):
    n = x.shape[0]
    flip = torch.rand(n, device=x.device) < 0.5
    x = torch.where(flip.view(-1, 1, 1, 1), x.flip(3), x)
    xp = F.pad(x, (4, 4, 4, 4), mode='reflect')
    i, j = torch.randint(0, 9, (2,))
    return xp[:, :, i:i + 32, j:j + 32]


@torch.no_grad()
def accuracy(model, x, y):
    model.eval()
    correct = sum((model(x[i:i + 1000]).argmax(1) == y[i:i + 1000]).sum().item() for i in range(0, len(x), 1000))
    model.train()
    return correct / len(x)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--arch', default='resnet18', choices=list(ARCHS))
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--bs', type=int, default=512)
    p.add_argument('--lr', type=float, default=0.1)
    p.add_argument('--data', default=os.path.expanduser('~/softvq_data/cifar-10-batches-py'))
    p.add_argument('--out', default=os.path.expanduser('~/softvq_runs/sim'))
    a = p.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = 'cuda'
    (xtr, ytr), (xte, yte) = load_cifar10(a.data, dev)
    model = ARCHS[a.arch]().to(dev)
    steps_per_epoch = len(xtr) // a.bs
    total = a.epochs * steps_per_epoch
    lr = a.lr if a.arch == 'resnet18' else a.lr / 5
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=total, pct_start=0.15)
    save_at = sorted({0, 10, 30, 100, 300, 1000, total})

    out = os.path.join(a.out, f'{a.arch}-s{a.seed}')
    os.makedirs(out, exist_ok=True)
    log = []
    step = 0
    model.train()
    for epoch in range(a.epochs):
        perm = torch.randperm(len(xtr), device=dev)
        for b in range(steps_per_epoch):
            if step in save_at:
                acc = accuracy(model, xte, yte)
                torch.save(model.state_dict(), os.path.join(out, f'step{step}.pt'))
                log.append({'step': step, 'test_acc': acc})
                print(f'step {step} test_acc {acc:.4f}', flush=True)
            idx = perm[b * a.bs:(b + 1) * a.bs]
            loss = F.cross_entropy(model(augment(xtr[idx])), ytr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            step += 1
    acc = accuracy(model, xte, yte)
    torch.save(model.state_dict(), os.path.join(out, f'step{step}.pt'))
    log.append({'step': step, 'test_acc': acc, 'train_acc': accuracy(model, xtr[:10000], ytr[:10000])})
    print(f'final step {step} test_acc {acc:.4f}', flush=True)
    import json
    with open(os.path.join(out, 'log.json'), 'w') as f:
        json.dump(log, f, indent=1)


if __name__ == '__main__':
    main()
