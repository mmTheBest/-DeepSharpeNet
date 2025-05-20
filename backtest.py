from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

def generate_signals(y_pred: np.ndarray, buy_thresh: float=0.0, sell_thresh: float=0.0):
    sig = np.zeros_like(y_pred, dtype=int)
    sig[y_pred > buy_thresh] = 1
    sig[y_pred < sell_thresh] = -1
    return sig

def backtest(returns: np.ndarray, signal: np.ndarray, starting_cap: float=1.0):
    strat_ret = signal * returns
    equity = starting_cap * np.cumprod(1 + strat_ret)
    total_ret = equity[-1] / equity[0] - 1.0
    daily_ret = strat_ret
    sharpe = np.sqrt(252) * daily_ret.mean() / (daily_ret.std() + 1e-08)
    running_max = np.maximum.accumulate(equity)
    drawdown = 1.0 - equity / running_max
    max_dd = drawdown.max()
    hold_equity = starting_cap * np.cumprod(1 + returns)
    hold_ret = hold_equity[-1] / hold_equity[0] - 1.0
    return {'equity': equity, 'hold_equity': hold_equity, 'total_return': total_ret, 'buy_hold_return': hold_ret, 'sharpe': sharpe, 'max_drawdown': max_dd}

def main():
    p = argparse.ArgumentParser()
    p.add_argument('pred_csv', type=Path, help='CSV with y_true,y_pred columns')
    p.add_argument('--ckpt', type=Path, default=Path('model_best.pt'), help='checkpoint file with normalization stats')
    p.add_argument('--buy', type=float, default=0.0, help='threshold above which to go long')
    p.add_argument('--sell', type=float, default=0.0, help='threshold below which to go short')
    p.add_argument('--outdir', type=Path, default=Path('bt_out'))
    args = p.parse_args()
    args.outdir.mkdir(exist_ok=True)
    parts = args.pred_csv.stem.split('_')
    code = parts[0]
    model_dir = Path('model_checkpoint')
    args.ckpt = model_dir / f'{code}.pt'
    df_pred = pd.read_csv(args.pred_csv)
    y_pred = df_pred['y_pred'].to_numpy()
    processed_csv = Path('processed_data') / f'{code}.csv'
    df_data = pd.read_csv(processed_csv)
    returns = df_data['ret'].to_numpy()
    if len(returns) > len(df_pred):
        returns = returns[-len(df_pred):]
    elif len(returns) < len(df_pred):
        raise ValueError(f'Mismatch between returns length {len(returns)} and predictions {len(df_pred)}')
    signal = generate_signals(y_pred, args.buy, args.sell)
    result = backtest(returns, signal)
    print('Total return: {0:.2%}'.format(result['total_return']))
    print('Buy & Hold return: {0:.2%}'.format(result['buy_hold_return']))
    print('Sharpe ratio: {0:.2f}'.format(result['sharpe']))
    print('Max drawdown: {0:.2%}'.format(result['max_drawdown']))
    plt.figure(figsize=(8, 4))
    plt.plot(result['equity'], label='Strategy equity')
    plt.plot(result['hold_equity'], label='Buy & Hold equity')
    plt.title('Strategy vs. Buy & Hold Equity')
    plt.legend()
    fig_path = args.outdir / f'{code}_equity.png'
    plt.tight_layout()
    plt.savefig(fig_path, dpi=120)
    print(f'Equity curve saved to {fig_path}')
    results_df = pd.DataFrame({'equity': result['equity'], 'buy_hold_equity': result['hold_equity'], 'signal': signal, 'returns': returns, 'strategy_returns': signal * returns})
    csv_path = args.outdir / f'{code}_backtest_results.csv'
    results_df.to_csv(csv_path, index=False)
    print(f'Backtest results saved to {csv_path}')
    metrics = {k: v for (k, v) in result.items() if k not in ['equity', 'hold_equity']}
    metrics_df = pd.DataFrame([metrics])
    metrics_path = args.outdir / f'{code}_backtest_metrics.csv'
    metrics_df.to_csv(metrics_path, index=False)
    print(f'Metrics saved to {metrics_path}')
if __name__ == '__main__':
    main()