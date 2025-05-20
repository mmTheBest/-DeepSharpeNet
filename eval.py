from __future__ import annotations
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import pandas as pd
from dataset import SlidingWindowDataset
from model import build_model

def load_model(ckpt_path, X, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    task = ckpt.get('task', 'reg')
    stats = ckpt.get('stats', None)
    selected_features = ckpt.get('selected_features', None)
    output_dim = 1 if task == 'reg' else 3
    model = build_model(X, task=task, output_dim=output_dim).to(device)
    model.load_state_dict(ckpt['model_state'], strict=False)
    model.eval()
    return (model, stats, task, selected_features)

def regression_metrics(y_true, y_pred):
    mse = F.mse_loss(y_pred, y_true).item()
    mae = F.l1_loss(y_pred, y_true).item()
    direction = (y_true.sign() == y_pred.sign()).float().mean().item()
    return (mse, mae, direction)

def classification_metrics(y_true, logits):
    probs = logits.softmax(dim=1)
    pred = probs.argmax(dim=1)
    acc = (pred == y_true).float().mean().item()
    return acc

def main():
    p = argparse.ArgumentParser()
    p.add_argument('csv', type=Path)
    p.add_argument('ckpt', type=Path)
    p.add_argument('--win', type=int, default=30)
    p.add_argument('--batch', type=int, default=128)
    p.add_argument('--outdir', type=Path, default=Path('eval_out'))
    args = p.parse_args()
    args.outdir.mkdir(exist_ok=True)
    parts = args.csv.stem.split('.')
    code = parts[1] if len(parts) > 1 else args.csv.stem
    tmp = pd.read_csv(args.csv, nrows=1)
    
    # Check for risk_adjusted_return first, then fall back to action columns if needed
    if 'risk_adjusted_return' in tmp.columns:
        label_col = 'risk_adjusted_return'
        print(f"[+] Using risk_adjusted_return as target for evaluation")
    else:
        action_cols = [c for c in tmp.columns if c.startswith('action_order')]
        if not action_cols:
            raise ValueError('No action_order column or risk_adjusted_return found in CSV for evaluation')
        label_col = action_cols[0]
        print(f"[+] Using {label_col} as target for evaluation")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(args.ckpt, map_location=device)
    selected_features = ckpt.get('selected_features', None)
    dummy_ds = SlidingWindowDataset(args.csv, win=args.win, normalize=False, label_col=label_col, features=selected_features)
    dummy_loader = DataLoader(dummy_ds, batch_size=1)
    batch = next(iter(dummy_loader))
    if len(batch) == 3:
        (sample_X, _, _) = batch
    else:
        (sample_X, _) = batch
    (model, stats, task, _) = load_model(args.ckpt, sample_X, device)
    ds = SlidingWindowDataset(args.csv, normalize=True, win=args.win, label_col=label_col, stats=stats, features=selected_features)
    total_samples = len(ds)
    test_size = 300
    if test_size > total_samples:
        raise ValueError(f'Not enough samples ({total_samples}) for test_size={test_size}')
    test_indices = list(range(total_samples - test_size, total_samples))
    test_ds = Subset(ds, test_indices)
    loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False)
    (all_pred, all_true) = ([], [])
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                (X, y, _) = batch  # Ignore distance label
            else:
                (X, y) = batch
            X = X.to(device)
            y = y.to(device)
            pred = model(X)  # Now returns a single output
            all_pred.append(pred.cpu())
            all_true.append(y.cpu())
    y_pred = torch.cat(all_pred)
    y_true = torch.cat(all_true)
    if task == 'reg':
        (mse, mae, direction) = regression_metrics(y_true, y_pred)
        rmse = mse ** 0.5
        ss_res = ((y_true - y_pred) ** 2).sum().item()
        sst = ((y_true - y_true.mean()) ** 2).sum().item()
        r2 = 1 - ss_res / sst if sst != 0 else float('nan')
        print(f'MSE {mse:.5f} | RMSE {rmse:.5f} | MAE {mae:.5f} | R2 {r2:.5f} | Directional acc {direction * 100:.2f}%')
        metrics = {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2, 'directional_accuracy': direction}
    else:
        acc = classification_metrics(y_true.long(), y_pred)
        print(f'Accuracy {acc * 100:.2f}%')
        metrics = {'accuracy': acc}
    df_out = pd.DataFrame({'y_true': y_true.numpy().squeeze(), 'y_pred': y_pred.numpy().squeeze()})
    csv_path = args.outdir / f'{code}_predictions.csv'
    df_out.to_csv(csv_path, index=False)
    print(f'Predictions stored in {csv_path}')
    metrics_df = pd.DataFrame([metrics])
    metrics_path = args.outdir / f'{code}_metrics.csv'
    metrics_df.to_csv(metrics_path, index=False)
    print(f'Metrics stored in {metrics_path}')
if __name__ == '__main__':
    main()