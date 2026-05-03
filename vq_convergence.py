import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torchvision
from tqdm import tqdm

try:
    import wandb as wb
except ImportError:
    wb = None

from utils import InputHook

def get_parser():
    parser = argparse.ArgumentParser(description="Train MLP on MNIST")
    
    # Experiment Settings
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--use_wandb', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--project_name', type=str, default='vqk_mnist', help='WandB project name')
    parser.add_argument('--run_name', type=str, default=None, help='Custom run name (optional)')
    
    # Training Hyperparameters
    parser.add_argument('--batch_size', type=int, default=196)
    parser.add_argument('--steps', type=int, default=50000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    
    # Model Architecture
    parser.add_argument('--width', type=int, default=200)
    parser.add_argument('--depth', type=int, default=4, help='Total number of layers (including output)')
    
    # System / I/O
    parser.add_argument('--num_logs', type=int, default=128, help='Number of log points (logarithmically spaced)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--data_dir', type=str, default='./data')
    
    return parser

def set_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_data(data_dir, batch_size):
    """Setup MNIST data loaders."""
    mnist_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        torchvision.transforms.Lambda(lambda x: x.flatten())
    ])
    
    train_ds = torchvision.datasets.MNIST(root=data_dir, train=True, transform=mnist_transform, download=True)
    test_ds = torchvision.datasets.MNIST(root=data_dir, train=False, transform=mnist_transform, download=True)

    # Pin memory speeds up host-to-device transfer
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    
    return train_loader, test_loader

def get_model(width, depth):
    """Create a scalable MLP."""
    # Input layer
    layers = [nn.Linear(784, width), nn.ReLU()]
    
    # Hidden layers
    for _ in range(depth - 2):
        layers.extend([nn.Linear(width, width), nn.ReLU()])
        
    # Output layer
    layers.append(nn.Linear(width, 10))
    
    return nn.Sequential(*layers)

def compute_accuracy(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, labels in loader:
            x = x.to(device)
            labels = labels.to(device)
            output = model(x)
            predicted_labels = torch.argmax(output, dim=1)
            correct += torch.sum(predicted_labels == labels).item()
            total += x.size(0)
            
    return correct / total

def remove_hooks(hooks):
    for hook in hooks.values():
        hook.remove()

def register_hooks(hooks):
    for hook in hooks.values():
        hook._register()

def clear_hooks(hooks):
    for hook in hooks.values():
        hook.clear()

def extract_vq_diffs(hooks):
    diffs={}
    hard_vqk=hooks[1.0].vq_kernel
    for b, h in hooks.items():
        if b==1.0: continue
        diffs[f'vqk/{b}']=(h.vq_kernel-hard_vqk).norm(p='fro').item()
    return diffs

def train(args):
    set_seeds(args.seed)
    
    # Determine run name
    run_name = args.run_name if args.run_name else f"{args.width}W-{args.seed}"

    # Initialize WandB conditionally
    if args.use_wandb:
        if wb is None:
            print("Warning: WandB requested but module not found. Install via 'pip install wandb'.")
        else:
            wb.init(project=args.project_name, config=vars(args), name=run_name)

    # Setup Data and Model
    train_loader, test_loader = get_data(args.data_dir, args.batch_size)
    loaders = {'train': train_loader, 'test': test_loader}
    
    model = get_model(args.width, args.depth).to(args.device)
    hooks = {b:InputHook(model, beta=b) for b in [0.6, 0.7, 0.8, 0.9, 1.0]}
    remove_hooks(hooks)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    one_hots = torch.eye(10, 10).to(args.device)
    base_mse = nn.MSELoss()
    criterion = lambda x, y: base_mse(x, one_hots[y])

    # Logging Steps Setup (Logarithmic spacing)
    logged_steps = np.unique(np.append(
        np.logspace(0, np.log10(args.steps), args.num_logs, dtype=int), 
        [0, args.steps]
    ))
    logged_steps = set(logged_steps)

    # Training Loop
    train_iter = iter(train_loader)
    pbar = tqdm(range(args.steps + 1), desc="Training")
    current_loss = 0.0

    for step in pbar:
        if step == args.steps:
            break

        if step in logged_steps:
            register_hooks(hooks)

        # --- Data Loading (Infinite Stream) ---
        try:
            x, labels = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, labels = next(train_iter)

        x = x.to(args.device)
        labels = labels.to(args.device)

        # --- Optimization ---
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        current_loss = loss.item()
        pbar.set_description(f'Loss: {current_loss:.4f}')

        if step in logged_steps:
            stats = {
                'step': step,
                **extract_vq_diffs(hooks),
                'loss/train': current_loss
                }
            remove_hooks(hooks)
            clear_hooks(hooks)

            stats['accuracy/test'] = compute_accuracy(model, loaders['test'], args.device)
            stats['accuracy/train'] = compute_accuracy(model, loaders['train'], args.device)
            
            # Switch back to train mode
            model.train() 

            if args.use_wandb and wb is not None and wb.run is not None:
                wb.log(stats)

    if args.use_wandb and wb is not None and wb.run is not None:
        wb.finish()

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    
    # Print configuration summary
    print("-" * 30)
    print("Running with configuration:")
    for key, val in vars(args).items():
        print(f"{key}: {val}")
    print("-" * 30)
    
    train(args)