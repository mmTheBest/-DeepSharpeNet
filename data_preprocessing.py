import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import index

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close = df['close'].astype(float)
    df['ret'] = close.pct_change()
    for win in (5, 10, 20, 30, 60):
        df[f'MA{win}'] = close.rolling(win).mean()
    for w in (6, 12, 24):
        index.calculate_RSI(df, w, column_name=f'RSI_{w}')
    for w in range(4, 37):
        index.calculate_wr(df, w, column_name=f'WR_{w}')
    for n in range(7, 37):
        index.calculate_kdj(df, n)
    for n in range(7, 37):
        index.calculate_bias(df, n, column_name=f'BIAS_{n}')
        index.calculate_psy(df, n, column_name=f'PSY_{n}')
    for n in range(7, 37):
        index.calculate_mfi(df, n, column_name=f'MFI_{n}')
        index.calculate_cci(df, n, column_name=f'CCI_{n}')
    for fast in range(4, 37, 1):
        slow = fast * 2 + 2
        index.calculate_MACD(df, (fast, slow), column_name=f'MACD_({fast},{slow})')
    for n in range(4, 37):
        df[f'PCTV_{n}'] = df['close'].pct_change(n) / df['close'].pct_change().rolling(n).std()
    for n in (14, 20, 24):
        atr_series = index.calculate_atr(df, n)
        df[f'ATR_{n}'] = atr_series
        index.calculate_roc(df, n)
    df.drop(columns=['ATR'], inplace=True, errors='ignore')
    index.calculate_obv(df)
    for n in (24,):
        index.calculate_vr(df, n, column_name=f'VR_{n}')
    for n in range(7, 37):
        dmi = index.calculate_dmi(df, n)
        df[f'+DI_{n}'] = dmi[f'+DI_{n}']
        df[f'-DI_{n}'] = dmi[f'-DI_{n}']
        df[f'ADX_{n}'] = dmi[f'ADX_{n}']
    index.calculate_sar(df)
    index.calculate_trix(df)
    index.calculate_bbi(df)
    index.calculate_obos(df)
    index.calculate_mass_index(df)
    drop_cols = ['HL', 'HC', 'LC', 'TR', '+DM', '-DM', 'TP', 'MA_TP', 'MD', 'N_days_up', 'PSY', 'High_Max', 'Low_Min', 'Range', 'EMA_Range', 'EMA_EMA_Range', 'Signal', 'Volume_Up', 'Volume_Down', 'Sum_Volume_Up', 'Sum_Volume_Down'] + [col for col in df.columns if col.startswith('DX_')]
    df.drop(columns=drop_cols, inplace=True, errors='ignore')
    return df

def label_extrema(df: pd.DataFrame, order: int=10) -> pd.DataFrame:
    close = df['close'].to_numpy()
    n = len(close)
    label = np.zeros(n, dtype=int)
    for i in range(order, n - order):
        left = close[i - order:i]
        right = close[i + 1:i + 1 + order]
        if close[i] > left.max() and close[i] > right.max():
            label[i] = 1
        elif close[i] < left.min() and close[i] < right.min():
            label[i] = -1
    df['extrema_label'] = label
    return df

def build_action(df: pd.DataFrame, n: int) -> pd.DataFrame:
    ext = df['extrema_label'].to_numpy()
    idxs = np.where(ext != 0)[0]
    action = np.zeros(len(df), dtype=float)
    sigma = n / 3.0
    for i in range(len(df)):
        if idxs.size == 0:
            continue
        nearest = idxs[np.argmin(np.abs(idxs - i))]
        d = abs(i - nearest)
        if d <= n:
            gaussian_factor = np.exp(-d ** 2 / (2 * sigma ** 2))
            action[i] = -ext[nearest] * gaussian_factor
    df[f'action_order{n}'] = action
    return df

def calculate_distance_to_extremum(df: pd.DataFrame, n: int) -> pd.DataFrame:
    ext = df['extrema_label'].to_numpy()
    idxs = np.where(ext != 0)[0]
    distance = np.zeros(len(df), dtype=float)
    for i in range(len(df)):
        if idxs.size == 0:
            distance[i] = n
            continue
        nearest = idxs[np.argmin(np.abs(idxs - i))]
        d = abs(i - nearest)
        distance[i] = min(d, n)
    df[f'distance_to_extremum'] = distance
    return df

def calculate_risk_adjusted_return(df: pd.DataFrame, k_bar: int=10, vol_window: int=k_bar) -> pd.DataFrame:
    """Calculate risk-adjusted k-bar return according to the formula:
       s_t^(k) = (C_{t+k} - C_t)/C_t / sqrt(1/k * sum_{i=1}^k r_{t+i}^2)
    """
    # Calculate forward k-bar return
    close = df['close']
    forward_return = (close.shift(-k_bar) - close) / close
    
    # Calculate returns for volatility
    returns = df['ret']
    
    # Initialize array for risk-adjusted returns
    risk_adjusted = np.zeros(len(df))
    
    # Compute for each point
    for t in range(len(df) - k_bar):
        # Get k future returns for volatility calculation
        future_returns = returns.iloc[t+1:t+k_bar+1].values
        
        # Skip if we don't have enough future data
        if len(future_returns) < k_bar:
            continue
            
        # Calculate denominator: sqrt(1/k * sum(r_{t+i}^2))
        future_vol = np.sqrt(np.mean(future_returns**2))
        
        # Avoid division by zero
        eps = 1e-8
        
        # Calculate risk-adjusted return
        if not np.isnan(forward_return.iloc[t]) and not np.isnan(future_vol):
            risk_adjusted[t] = forward_return.iloc[t] / (future_vol + eps)
    
    # Store in DataFrame
    df['risk_adjusted_return'] = risk_adjusted
    
    return df

def main():
    p = argparse.ArgumentParser()
    p.add_argument('excel', type=Path, help='input Excel file (OHLCV sheet)')
    p.add_argument('--sheet', default=0, help='sheet name/index holding k‑line')
    p.add_argument('--out', type=Path)
    p.add_argument('--order', type=int, default=10, help='extrema window size')
    p.add_argument('--k-bar', type=int, default=10, help='forward bars for risk-adjusted return')
    p.add_argument('--vol-window', type=int, default=20, help='volatility window size')
    args = p.parse_args()
    df = pd.read_excel(args.excel, sheet_name=args.sheet).rename(columns=str.lower)
    df = compute_indicators(df)
    df = label_extrema(df, args.order)
    df = build_action(df, args.order)
    df = calculate_distance_to_extremum(df, args.order)
    df = calculate_risk_adjusted_return(df, args.k_bar, args.vol_window)
    df = df.dropna().reset_index(drop=True)
    df.columns = df.columns.str.lower()
    out_dir = Path('processed_data')
    out_dir.mkdir(exist_ok=True)
    code = args.excel.stem.split('.')[-1]
    out_path = args.out if args.out else out_dir / f'{code}.csv'
    df.to_csv(out_path, index=False)
    print(f'[+] saved {out_path}')
if __name__ == '__main__':
    main()