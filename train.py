from __future__ import annotations
import argparse
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split, Subset
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torch.nn.functional as F
from dataset import SlidingWindowDataset
import pandas as pd
from model import build_model
import numpy as np

def split_dataset(ds: SlidingWindowDataset, val_ratio: float=0.1):
    n = len(ds)
    n_val = int(n * val_ratio)
    n_train = n - n_val
    gen = torch.Generator().manual_seed(42)
    return random_split(ds, [n_train, n_val], generator=gen)

def huber_weighted(pred, y, delta=1.0, eps=1e-08):
    error = torch.where((y - pred).abs() < delta, 0.5 * (y - pred).pow(2), delta * (y - pred).abs() - 0.5 * delta ** 2)
    w = 1.0 + y.abs()
    return (w * error).mean()

def calculate_group_loss(pred, y, lambda_group):
    return 0.0

def calculate_correlation_loss(pred, y, alpha_corr):
    if alpha_corr <= 0:
        return 0.0
    vx = pred - pred.mean()
    vy = y - y.mean()
    corr = (vx * vy).mean() / (vx.std() * vy.std() + 1e-08)
    return alpha_corr * (1 - corr)

def calculate_r2_loss(pred, y, gamma_r2):
    if gamma_r2 <= 0:
        return 0.0
    mse = F.mse_loss(pred, y)
    var_y = y.var(unbiased=False)
    r2_like = mse / (var_y + 1e-08)
    return gamma_r2 * r2_like

def calculate_sparse_loss(model, beta_sparse):
    if beta_sparse <= 0 or not hasattr(model, 'conv') or (not hasattr(model.conv[0], 'gate_conv')):
        return 0.0
    gate_weights = model.conv[0].gate_conv.weight
    feature_sparsity = gate_weights.abs().sum(dim=(0, 2)).mean()
    return beta_sparse * feature_sparsity

def train_epoch(model, loader, loss_fn, optimiser, device, lambda_group, alpha_corr, gamma_r2, beta_sparse):
    model.train()
    total_loss = 0.0
    for batch in loader:
        if len(batch) == 3:
            (X, y, _) = batch  # Ignore distance label
        else:
            (X, y) = batch
        (X, y) = (X.to(device), y.to(device))
        
        optimiser.zero_grad()
        pred = model(X)  # Now returns a single output
        
        mse_loss = loss_fn(pred, y)
        corr_loss = calculate_correlation_loss(pred, y, alpha_corr)
        r2_loss = calculate_r2_loss(pred, y, gamma_r2)
        sparse_loss = calculate_sparse_loss(model, beta_sparse)
        
        # Group lasso regularization
        if lambda_group > 0 and hasattr(model, 'conv1'):
            if hasattr(model.conv1[0], 'conv'):
                conv1_weight = model.conv1[0].conv.weight
                group_norms = conv1_weight.view(conv1_weight.size(0), -1).norm(p=2, dim=1)
                group_loss = lambda_group * group_norms.sum()
            elif hasattr(model.conv1[0], 'weight'):
                conv1_weight = model.conv1[0].weight
                group_norms = conv1_weight.view(conv1_weight.size(0), -1).norm(p=2, dim=1)
                group_loss = lambda_group * group_norms.sum()
            else:
                group_loss = 0.0
        else:
            group_loss = 0.0
        
        loss = mse_loss + corr_loss + r2_loss + sparse_loss + group_loss
        loss.backward()
        optimiser.step()
        total_loss += loss.item() * X.size(0)
    return total_loss / len(loader.dataset)

def eval_epoch(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                (X, y, _) = batch  # Ignore distance label
            else:
                (X, y) = batch
            (X, y) = (X.to(device), y.to(device))
            pred = model(X)  # Now returns a single output
            loss = loss_fn(pred, y)
            total_loss += loss.item() * X.size(0)
    return total_loss / len(loader.dataset)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('csv', type=Path, help='path to *_basic.csv produced earlier')
    p.add_argument('--win', type=int, default=30)
    p.add_argument('--target-shift', type=int, default=1, help='steps ahead to predict (default: 1)')
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--task', choices=['reg', 'cls'], default='reg')
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--patience', type=int, default=8, help='early‑stop patience')
    p.add_argument('--ckpt', type=Path, default='model_best.pt')
    p.add_argument('--lambda-group', type=float, default=0.0001, help='sparse-group lasso strength on first conv layer')
    p.add_argument('--alpha-corr', type=float, default=0.2, help='weight of (1-corr) term added to loss')
    p.add_argument('--gamma-r2', type=float, default=0.5, help='weight of variance-normalized MSE term (proxies −R2)')
    p.add_argument('--raw-only', action='store_true', help='use only the 9 base features: open,high,low,close,volume,ret,ma5,ma10,ma20')
    p.add_argument('--beta-sparse', type=float, default=0.001, help='weight for feature sparsity loss (L1 on gates)')
    p.add_argument('--balanced-sampling', action='store_true', help='use balanced sampling from each magnitude bin')
    args = p.parse_args()
    parts = args.csv.stem.split('.')
    code = parts[1] if len(parts) > 1 else args.csv.stem
    model_dir = Path('model_checkpoint')
    model_dir.mkdir(exist_ok=True)
    args.ckpt = model_dir / f'{code}.pt'
    tmp = pd.read_csv(args.csv, nrows=1)
    # Check for risk_adjusted_return first, then fall back to action columns if needed
    if 'risk_adjusted_return' in tmp.columns:
        label_col = 'risk_adjusted_return'
        print(f"[+] Using risk_adjusted_return as target")
    else:
        action_cols = [c for c in tmp.columns if c.startswith('action_order')]
        if not action_cols:
            raise ValueError('No action_order column or risk_adjusted_return found in processed CSV for training')
        label_col = action_cols[0]
        print(f"[+] Using {label_col} as target")
    args.task = 'reg'
    if args.raw_only:
        important_features = ['open', 'high', 'low', 'close', 'volume', 'ret', 'ma5', 'ma10', 'ma20']
        print(f'[+] Using raw-only features: {important_features}')
    else:
        important_features = None
        print(f'[+] Using all available features')
    base_ds = SlidingWindowDataset(args.csv, win=args.win, target_shift=args.target_shift, label_col=label_col, normalize=True, features=important_features)
    total_samples = len(base_ds)
    test_size = min(300, total_samples // 5)
    if test_size >= total_samples:
        raise ValueError(f'Not enough samples ({total_samples}) for test_size={test_size}')
    train_val_size = total_samples - test_size
    train_val_indices = list(range(train_val_size))
    train_val_ds = Subset(base_ds, train_val_indices)
    train_val_ds.stats = base_ds.stats
    (train_ds, val_ds) = split_dataset(train_val_ds, 0.1)
    val_ds.dataset.stats = train_ds.dataset.stats
    df_train = pd.DataFrame({'label': [base_ds.df.loc[train_val_indices[i] + base_ds.win + base_ds.shift - 1, label_col] for i in train_ds.indices if train_val_indices[i] + base_ds.win + base_ds.shift - 1 < len(base_ds.df)]})
    valid_indices = [idx for idx in train_ds.indices if train_val_indices[idx] + base_ds.win + base_ds.shift - 1 < len(base_ds.df)]
    hi = df_train['label'].abs() >= 0.6
    mid = (df_train['label'].abs() < 0.6) & (df_train['label'].abs() >= 0.2)
    lo = df_train['label'].abs() < 0.2
    total_samples = len(df_train)
    hi_count = hi.sum()
    mid_count = mid.sum()
    lo_count = lo.sum()
    print(f'\nTarget magnitude distribution in training set:')
    print(f'  High magnitude (|y| >= 0.6): {hi_count} samples ({hi_count / total_samples:.1%})')
    print(f'  Medium magnitude (0.2 <= |y| < 0.6): {mid_count} samples ({mid_count / total_samples:.1%})')
    print(f'  Low magnitude (|y| < 0.2): {lo_count} samples ({lo_count / total_samples:.1%})')
    samples_per_bin = min(hi_count, mid_count, lo_count)
    balanced_ratio = args.balanced_sampling
    if balanced_ratio:
        hi_indices = [valid_indices[i] for (i, is_hi) in enumerate(hi) if is_hi]
        mid_indices = [valid_indices[i] for (i, is_mid) in enumerate(mid) if is_mid]
        lo_indices = [valid_indices[i] for (i, is_lo) in enumerate(lo) if is_lo]
        np.random.seed(42)
        if len(hi_indices) > samples_per_bin:
            hi_indices = np.random.choice(hi_indices, samples_per_bin, replace=False)
        if len(mid_indices) > samples_per_bin:
            mid_indices = np.random.choice(mid_indices, samples_per_bin, replace=False)
        if len(lo_indices) > samples_per_bin:
            lo_indices = np.random.choice(lo_indices, samples_per_bin, replace=False)
        balanced_indices = np.concatenate([hi_indices, mid_indices, lo_indices])
        print(f'\nUsing balanced sampling with {samples_per_bin} samples per magnitude bin')
        sampler = torch.utils.data.SubsetRandomSampler(balanced_indices)
    else:
        hi_indices = [valid_indices[i] for (i, is_hi) in enumerate(hi) if is_hi]
        mid_indices = [valid_indices[i] for (i, is_mid) in enumerate(mid) if is_mid]
        lo_indices = [valid_indices[i] for (i, is_lo) in enumerate(lo) if is_lo]
        np.random.seed(42)
        hi_sample = np.random.choice(hi_indices, size=min(int(len(hi_indices) * 0.5), len(hi_indices)), replace=False)
        mid_sample = np.random.choice(mid_indices, size=min(int(len(mid_indices) * 0.3), len(mid_indices)), replace=False)
        lo_sample = np.random.choice(lo_indices, size=min(int(len(lo_indices) * 0.2), len(lo_indices)), replace=False)
        sampler_indices = np.concatenate([hi_sample, mid_sample, lo_sample])
        print(f'\nUsing original sampling with fixed fractions (50%/30%/20%)')
        sampler = torch.utils.data.SubsetRandomSampler(sampler_indices)
    train_loader = DataLoader(train_ds.dataset, batch_size=args.batch, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    (sample_X, _, _) = next(iter(train_loader))
    model = build_model(sample_X, task=args.task)
    model.to(device)
    if args.task == 'reg':
        loss_fn = nn.SmoothL1Loss()
    else:
        loss_fn = nn.CrossEntropyLoss()
    optimiser = Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingWarmRestarts(optimiser, T_0=10, T_mult=1)
    lambda_group = args.lambda_group
    train_losses = []
    val_losses = []
    best_loss = float('inf')
    patience_left = args.patience
    for epoch in range(1, args.epochs + 1):
        tr_loss = train_epoch(model, train_loader, loss_fn, optimiser, device, lambda_group, args.alpha_corr, args.gamma_r2, args.beta_sparse)
        val_loss = eval_epoch(model, val_loader, loss_fn, device)
        scheduler.step()
        train_losses.append(tr_loss)
        val_losses.append(val_loss)
        print(f'Epoch {epoch:02d} | train {tr_loss:.4f} | val {val_loss:.4f}')
        if val_loss < best_loss - 0.0001:
            best_loss = val_loss
            torch.save({'model_state': model.state_dict(), 'stats': train_ds.dataset.stats, 'task': args.task, 'selected_features': important_features}, args.ckpt)
            patience_left = args.patience
            print('  ↳ new best; checkpoint saved.')
        else:
            patience_left -= 1
            if patience_left == 0:
                print('Early stopping triggered.')
                break
    print(f'Training done; best val loss = {best_loss:.4f}')
if __name__ == '__main__':
    main()