import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import argrelextrema
from scipy.stats import zscore

def calculate_macd1(df):
    ema3 = df['close'].ewm(span=3, adjust=False).mean()
    ema5 = df['close'].ewm(span=5, adjust=False).mean()
    macd1 = ema3 - ema5
    signal_line1 = macd1.ewm(span=4, adjust=False).mean()
    histogram1 = macd1 - signal_line1
    sign_changes = np.sign(histogram1).diff()
    zero_crossings = np.zeros_like(histogram1)
    zero_crossings[sign_changes == 2] = 1
    zero_crossings[sign_changes == -2] = -1
    MACD1 = pd.Series(zero_crossings, index=histogram1.index)
    return (macd1, signal_line1, histogram1, MACD1)

def calculate_macd2(df):
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    macd2 = ema12 - ema26
    signal_line2 = macd2.ewm(span=9, adjust=False).mean()
    histogram2 = macd2 - signal_line2
    sign_changes = np.sign(histogram2).diff()
    zero_crossings = np.zeros_like(histogram2)
    zero_crossings[sign_changes == 2] = 1
    zero_crossings[sign_changes == -2] = -1
    MACD2 = pd.Series(zero_crossings, index=histogram2.index)
    return (macd2, signal_line2, histogram2, MACD2)

def calculate_macd_with_variance(df, short_period, long_period, signal_period, column='close'):

    def calculate_weighted_ema(df, m, column):
        sigma = df[column].rolling(window=m, min_periods=1).std()
        weighted_ema = np.zeros(len(df))
        weighted_ema[0] = df[column].iloc[0]
        for t in range(1, len(df)):
            window_start = max(0, t - m)
            sigma_sum_past = sigma.iloc[window_start:t].sum()
            sigma_sum_all = sigma_sum_past + sigma.iloc[t]
            if sigma_sum_all == 0:
                continue
            weighted_ema[t] = sigma_sum_past / sigma_sum_all * weighted_ema[t - 1] + sigma.iloc[t] / sigma_sum_all * df[column].iloc[t]
        return weighted_ema
    short_ema = calculate_weighted_ema(df, short_period, column)
    long_ema = calculate_weighted_ema(df, long_period, column)
    DIF = short_ema - long_ema
    DEA = calculate_weighted_ema(df, signal_period, DIF)
    OSC = DIF - DEA
    osc_series = pd.Series(OSC, index=df.index)
    sign_changes = osc_series.diff()
    zero_crossings = np.zeros_like(OSC)
    zero_crossings[sign_changes == 2] = 1
    zero_crossings[sign_changes == -2] = -1
    MACD_var = pd.Series(zero_crossings, index=osc_series.index)
    df['OSC'] = osc_series
    df['MACD_var'] = MACD_var
    return df[['OSC', 'MACD_var']]

def calculate_rsi(df, window=14):
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss.where(loss != 0, pd.NA)
    rs.fillna(0, inplace=True)
    rsi = 100 - 100 / (1 + rs)
    RSI_signal = pd.Series(0, index=df.index)
    RSI_signal[(rsi > 70) & (rsi.shift(1) <= 70)] = -1
    RSI_signal[(rsi < 30) & (rsi.shift(1) >= 30)] = 1
    df['rsi'] = rsi
    df['rsi_signal'] = RSI_signal
    return df[['rsi', 'rsi_signal']]

def calculate_kdj_yuan(df, n):
    required_columns = ['high', 'low', 'close']
    for col in required_columns:
    low_n = df['low'].rolling(window=n, min_periods=1).min()
    high_n = df['high'].rolling(window=n, min_periods=1).max()
    rsv = (df['close'] - low_n) / (high_n - low_n) * 100
    k_values = np.zeros(len(df))
    d_values = np.zeros(len(df))
    k_values[0] = 50
    d_values[0] = 50
    for i in range(1, len(df)):
        if high_n.iloc[i] == low_n.iloc[i]:
            rsv.iloc[i] = np.nan
        else:
            rsv.iloc[i] = (df['close'].iloc[i] - low_n.iloc[i]) / (high_n.iloc[i] - low_n.iloc[i]) * 100
        if pd.isna(rsv.iloc[i]):
            rsv.iloc[i] = k_values[i - 1]
        k_values[i] = 2 / 3 * k_values[i - 1] + 1 / 3 * rsv.iloc[i]
        d_values[i] = 2 / 3 * d_values[i - 1] + 1 / 3 * k_values[i]
    j_values = 3 * k_values - 2 * d_values
    df['K'] = k_values
    df['D'] = d_values
    df['J'] = j_values
    kdj_signal = pd.Series(0, index=df.index)
    kdj_signal[(df['K'] > df['D']) & (df['K'].shift(1) <= df['D'].shift(1)) & (df['J'] < 0)] = 1
    kdj_signal[(df['K'] < df['D']) & (df['K'].shift(1) >= df['D'].shift(1)) & (df['J'] > 100)] = -1
    df['kdj_signal'] = kdj_signal
    return df[['K', 'D', 'J', 'kdj_signal']]

def calculate_vwap(df):
    price_volume = df['close'] * df['volume']
    vwap = price_volume.cumsum() / df['volume'].cumsum()
    return vwap

def calculate_bollinger_bands(df, window=20):
    rolling_mean = df['close'].rolling(window).mean()
    rolling_std = df['close'].rolling(window).std()
    upper_band = rolling_mean + rolling_std * 2
    lower_band = rolling_mean - rolling_std * 2
    bollinger_signal = pd.Series(index=df.index, dtype=float)
    bollinger_signal[df['close'] > upper_band] = -1
    bollinger_signal[df['close'] < lower_band] = 1
    bollinger_signal[(df['close'] <= upper_band) & (df['close'] >= lower_band)] = 0
    return bollinger_signal

def calculate_relative_coefficients(df):
    df['high_low_Coef'] = (df['high'] - df['low']) / df['low']
    df['open_close_coef'] = (df['close'] - df['open']) / df['open']
    return df[['high_low_Coef', 'open_close_coef']]

def calculate_asi(df, period=14):
    df['Swing'] = (df['close'] - df['close'].shift(1)) / (df['high'] - df['low'])
    df['Sum_Swing'] = df['Swing'].rolling(window=period).sum()
    df['ASI'] = df['Sum_Swing'].rolling(window=period).mean()
    return df[['ASI']]

def calculate_arbr(df, period=26):
    df['AR'] = df['high'].rolling(window=period).apply(lambda x: sum(x - df['open'][x.index]), raw=False) / df['low'].rolling(window=period).apply(lambda x: sum(df['open'][x.index] - x), raw=False) * 100
    df['BR'] = df['close'].rolling(window=period).apply(lambda x: sum(x - df['open'].shift(1)[x.index]), raw=False) / df['open'].rolling(window=period).apply(lambda x: sum(x - df['close'].shift(1)[x.index]), raw=False) * 100
    return df[['AR', 'BR']]

def calculate_dpo(df, period=20):
    offset = int(period / 2 + 1)
    df['MA'] = df['close'].rolling(window=period).mean()
    df['DPO'] = df['close'] - df['MA'].shift(offset)
    return df[['DPO']]

def calculate_trix(df, period=15):
    ema1 = df['close'].ewm(span=period, adjust=False).mean()
    ema2 = ema1.ewm(span=period, adjust=False).mean()
    ema3 = ema2.ewm(span=period, adjust=False).mean()
    df['TRIX'] = (ema3 - ema3.shift(1)) / ema3.shift(1) * 100
    return df[['TRIX']]

def calculate_bbi(df):
    sma_3 = df['close'].rolling(window=3).mean()
    sma_6 = df['close'].rolling(window=6).mean()
    sma_12 = df['close'].rolling(window=12).mean()
    sma_24 = df['close'].rolling(window=24).mean()
    df['BBI'] = (sma_3 + sma_6 + sma_12 + sma_24) / 4
    return df[['BBI']]

def calculate_mtm(df, period=12):
    df['MTM'] = df['close'] - df['close'].shift(period)
    return df

def calculate_obv(df):
    obv = [0]
    for i in range(1, len(df)):
        if df['close'].iloc[i] > df['close'].iloc[i - 1]:
            obv.append(obv[-1] + df['volume'].iloc[i])
        elif df['close'].iloc[i] < df['close'].iloc[i - 1]:
            obv.append(obv[-1] - df['volume'].iloc[i])
        else:
            obv.append(obv[-1])
    df['OBV'] = obv
    return df[['OBV']]

def calculate_sar(df, af=0.02, af_step=0.02, af_max=0.2):
    sar = df['low'][0]
    ep = df['high'][0]
    trend = 1
    af_current = af
    sar_list = [sar]
    for i in range(1, len(df)):
        sar = sar + af_current * (ep - sar)
        if trend == 1:
            if df['low'][i] < sar:
                trend = -1
                sar = ep
                ep = df['low'][i]
                af_current = af
            elif df['high'][i] > ep:
                ep = df['high'][i]
                af_current = min(af_current + af_step, af_max)
        elif trend == -1:
            if df['high'][i] > sar:
                trend = 1
                sar = ep
                ep = df['high'][i]
                af_current = af
            elif df['low'][i] < ep:
                ep = df['low'][i]
                af_current = min(af_current + af_step, af_max)
        sar_list.append(sar)
    df['SAR'] = sar_list
    return df[['SAR']]

def calculate_ema(df, period=20):
    df['EMA'] = df['close'].ewm(span=period, adjust=False).mean()
    return df

def calculate_obos(df, period=14):
    df['High_Max'] = df['high'].rolling(window=period).max()
    df['Low_Min'] = df['low'].rolling(window=period).min()
    df['OBOS'] = 100 * (df['close'] - df['Low_Min']) / (df['High_Max'] - df['Low_Min'])
    return df[['OBOS']]

def calculate_sma(df, window=14):
    df['SMA'] = df['close'].rolling(window=window).mean()
    return df

def calculate_wma(df, window=14):
    weights = pd.Series(range(1, window + 1))
    df['WMA'] = df['close'].rolling(window=window).apply(lambda prices: (prices * weights).sum() / weights.sum(), raw=True)
    return df[['WMA']]

def calculate_mass_index(df, period=9, threshold=27):
    df['Range'] = df['high'] - df['low']
    df['EMA_Range'] = df['Range'].ewm(span=period, adjust=False).mean()
    df['EMA_EMA_Range'] = df['EMA_Range'].ewm(span=period, adjust=False).mean()
    df['Mass Index'] = df['EMA_EMA_Range'] / df['EMA_Range']
    df['Signal'] = np.where(df['Mass Index'] > threshold, 1, 0)
    return df[['Mass Index', 'Signal']]

def calculate_roc(df, period=14):
    df[f'ROC_{period}'] = (df['close'] - df['close'].shift(period)) / df['close'].shift(period) * 100
    return df

def calculate_adx(df, period=14):
    df['HL'] = df['high'] - df['low']
    df['HC'] = abs(df['high'] - df['close'].shift(1))
    df['LC'] = abs(df['low'] - df['close'].shift(1))
    df['TR'] = df[['HL', 'HC', 'LC']].max(axis=1)
    high_diff = df['high'] - df['high'].shift(1)
    low_diff = df['low'].shift(1) - df['low']
    df['+DM'] = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
    df['-DM'] = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
    df['+DM_Sum'] = df['+DM'].rolling(window=period).sum()
    df['-DM_Sum'] = df['-DM'].rolling(window=period).sum()
    df['TR_Sum'] = df['TR'].rolling(window=period).sum()
    df['+DI'] = 100 * (df['+DM_Sum'] / (df['TR_Sum'] + 1e-10))
    df['-DI'] = 100 * (df['-DM_Sum'] / (df['TR_Sum'] + 1e-10))
    df['DX'] = 100 * abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'] + 1e-10)
    df['ADX'] = df['DX'].rolling(window=period).mean()
    return df[['+DI', '-DI', 'ADX']]

def calculate_atr(df, period=14):
    if not all((col in df.columns for col in ['high', 'low', 'close'])):
        raise ValueError("DataFrame must contain 'high', 'low', and 'close' columns")
    df['HL'] = df['high'] - df['low']
    df['HC'] = abs(df['high'] - df['close'].shift(1))
    df['LC'] = abs(df['low'] - df['close'].shift(1))
    df['TR'] = df[['HL', 'HC', 'LC']].max(axis=1)
    df['ATR'] = df['TR'].rolling(window=period, min_periods=1).mean()
    return df[['ATR']]

def find_local_extrema(close_prices, order=5):
    max_idx = argrelextrema(close_prices.values, np.greater, order=order)[0]
    min_idx = argrelextrema(close_prices.values, np.less, order=order)[0]
    return (max_idx, min_idx)

def calculate_action(df, order=5):
    close_prices = df['close']
    (max_idx, min_idx) = find_local_extrema(close_prices, order=order)
    df['action_+t'] = np.nan
    df['action_-t'] = np.nan
    df['H_N'] = np.nan
    df['I_N'] = np.nan
    df.loc[max_idx, 'H_N'] = df.loc[max_idx, 'close']
    df.loc[min_idx, 'I_N'] = df.loc[min_idx, 'close']
    df['H_N'].ffill(inplace=True)
    df['I_N'].ffill(inplace=True)
    df['action_+t'] = np.abs((df['close'] - df['H_N']) / (df['H_N'] - df['I_N']))
    df['action_-t'] = np.abs((df['close'] - df['I_N']) / (df['H_N'] - df['I_N']))
    return df[['action_+t', 'action_-t', 'H_N', 'I_N']]

def calculate_time_gap_from_sheets(df, resolution):
    if resolution in ['d_kline', 'w_kline', 'm_kline']:
        time_col = 'date'
        print(df[time_col].dtype)
        df[time_col] = pd.to_datetime(df[time_col])
        df['time_diff'] = (df[time_col] - df[time_col].shift()).dt.days.fillna(0)
    elif resolution in ['5_kline', '15_kline', '30_kline', '60_kline']:
        time_col = 'time'
        print(df[time_col].dtype)
        df['time_diff'] = df[time_col] - df[time_col].shift().fillna(0)
    else:
        raise ValueError(f'Unsupported resolution: {resolution}')
    if resolution in ['d_kline', 'w_kline', 'm_kline']:
        df['time_gap'] = df['time_diff']
    elif resolution in ['5_kline', '15_kline', '30_kline', '60_kline']:
        multiplier = {'5_kline': 500000, '15_kline': 1500000, '30_kline': 3000000, '60_kline': 6000000}
        df['time_gap'] = df['time_diff'] / multiplier[resolution]
    else:
        raise ValueError(f'Unsupported resolution: {resolution}')
    return df

def remove_extreme_values(df, threshold=2):
    if 'AR' not in df.columns or 'BR' not in df.columns:
        raise ValueError("DataFrame must contain 'AR' and 'BR' columns")
    df['z_score_AR'] = zscore(df['AR'])
    df['z_score_BR'] = zscore(df['BR'])
    extreme_conditions = (np.abs(df['z_score_AR']) > threshold) | (np.abs(df['z_score_BR']) > threshold) | np.isinf(df['AR']) | np.isinf(df['BR'])
    extreme_indices = df[extreme_conditions].index
    cleaned_df = df.drop(extreme_indices)
    cleaned_df.drop(columns=['z_score_AR', 'z_score_BR'], inplace=True)
    return cleaned_df

def classify_action(df, threshold=0.05, action_name='action_order10'):
    df['action'] = np.where(pd.isnull(df[action_name]), 'n', np.where((df[action_name] != 'n') & ((df[action_name] >= 1 - threshold) & (df[action_name] <= 1)), 1, np.where((df[action_name] != 'n') & ((df[action_name] <= -1 + threshold) & (df[action_name] >= -1)), -1, 0)))
    return df

def calculate_close_comparison_indicator(df, n=1):
    df = df.sort_index()
    indicator_column_name = f'indicator_{n}'
    df[indicator_column_name] = np.nan
    for i in range(len(df) - n):
        if df['close'].iloc[i] > df['close'].iloc[i + n]:
            df.loc[i, indicator_column_name] = -1
        else:
            df.loc[i, indicator_column_name] = 1
    return df[[indicator_column_name]]

def calculate_close_rate_of_change(df, n=1):
    df = df.sort_index()
    rate_of_change_column_name = f'rate_of_close_change_{n}'
    df[rate_of_change_column_name] = np.nan
    for i in range(len(df) - n):
        current_close = df['close'].iloc[i]
        future_close = df['close'].iloc[i + n]
        rate_of_change = (future_close - current_close) / current_close
        df.loc[i, rate_of_change_column_name] = rate_of_change
    return df[[rate_of_change_column_name]]

def calculate_volume_rate_of_change(df, n=1):
    df = df.sort_index()
    rate_of_change_column_name = f'rate_of_volume_change_{n}'
    df[rate_of_change_column_name] = np.nan
    for i in range(len(df) - n):
        current_close = df['volume'].iloc[i]
        future_close = df['volume'].iloc[i + n]
        rate_of_change = (future_close - current_close) / current_close
        df.loc[i, rate_of_change_column_name] = rate_of_change
    return df[[rate_of_change_column_name]]

def log_return(df, price_col='close'):
    df['log_return'] = np.log(df[price_col] / df[price_col].shift(1))
    return df

def log_volume_change(df, volume_col='volume'):
    df['log_volume_change'] = np.log(df[volume_col] / df[volume_col].shift(1))
    return df

def volatility(df, price_col='close', window=20):
    df['volatility'] = df[price_col].pct_change().rolling(window=window).std()
    return df

def fill_turn_na(df, turn_col='turn'):
    if turn_col in df.columns:
        df[turn_col] = df[turn_col].fillna(0)
    return df

def ATP(df):
    atp = df['amount'] / df['volume']
    atp[df['volume'] == 0] = 0
    df['ATP'] = atp
    return df['ATP']

def ALT(df):
    ALT = (df['high'] - df['low']) / df['open']
    df['ALT'] = ALT
    return df['ALT']

def ITL(df):
    ITL = (df['close'] > df['open']).astype(int)
    df['ITL'] = ITL
    return df['ITL']

def calculate_CATP(df):
    df['CATP'] = (df['ATP'] - df['ATP'].shift(1)) / df['ATP'].shift(1)
    return df['CATP']

def CTM(df):
    CTM = (df['amount'] - df['amount'].shift(1)) / df['amount'].shift(1)
    df['CTM'] = CTM
    return df['CTM']

def CTR(df):
        return None
    df['CTR'] = (df['turn'] - df['turn'].shift(1)) / df['turn'].shift(1)
    return df['CTR']

def calculate_PCCP(df):
    required_columns = ['high', 'low', 'close']
    pccp = df.apply(lambda row: (2 * row['close'] - row['low'] - row['high']) / (row['high'] - row['low']) if row['high'] != row['low'] else 1, axis=1)
    df['PCCP'] = pccp
    return df['PCCP']

def calculate_RDMA(df):
    ma10 = df['close'].rolling(window=10).mean()
    ma50 = df['close'].rolling(window=50).mean()
    rdma = np.where(ma50 != 0, (ma10 - ma50) / ma50, np.nan)
    df['RDMA'] = rdma
    return df['RDMA']

def calculate_rmacd(df):
    ma12 = df['close'].rolling(window=12).mean()
    ma26 = df['close'].rolling(window=26).mean()
    rdiff = ma12 - ma26
    rdiff_9 = rdiff.rolling(window=9).mean()
    rmacd = rdiff - rdiff_9
    df['RMACD'] = rmacd
    return df['RMACD']

def calculate_RSI(df, window, column_name):
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss.where(loss != 0, pd.NA)
    rs.fillna(0, inplace=True)
    rsi = 100 - 100 / (1 + rs)
    df[column_name] = rsi
    return df[column_name]

def calculate_wr(df, period, column_name):
    high_max = df['high'].rolling(window=period).max()
    low_min = df['low'].rolling(window=period).min()
    WR = (high_max - df['close']) / (high_max - low_min)
    df[column_name] = WR
    return df[column_name]

def calculate_MACD(df, periods, column_name):
    if isinstance(periods, (tuple, list)) and len(periods) == 2:
        (fast, slow) = periods
    else:
        (fast, slow) = (periods, 2 * periods + 2)
    alpha_fast = 2 / (fast + 1)
    alpha_slow = 2 / (slow + 1)
    ema_fast = df['close'].ewm(alpha=alpha_fast, adjust=False).mean()
    ema_slow = df['close'].ewm(alpha=alpha_slow, adjust=False).mean()
    df[column_name] = ema_fast - ema_slow
    return df[column_name]

def calculate_PCTV(df, window, column_name):
    tv_max = np.full(len(df), np.nan)
    tv_min = np.full(len(df), np.nan)
    for i in range(window, len(df)):
        tv_max[i] = df['volume'][i - window + 1:i + 1].max()
        tv_min[i] = df['volume'][i - window + 1:i + 1].min()
    tv_max = pd.Series(tv_max, index=df.index)
    tv_min = pd.Series(tv_min, index=df.index)
    pctv = np.where((tv_max != tv_min) & ~np.isnan(tv_max) & ~np.isnan(tv_min), (2 * df['volume'] - tv_min - tv_max) / (tv_max - tv_min), 1)
    df[column_name] = pctv
    return df[column_name]

def calculate_kdj(df, n):
    required_columns = ['high', 'low', 'close']
    for col in required_columns:
    low_n = df['low'].rolling(window=n, min_periods=1).min()
    high_n = df['high'].rolling(window=n, min_periods=1).max()
    rsv = (df['close'] - low_n) / (high_n - low_n) * 100
    rsv[df['high'] == df['low']] = 50
    k_values = np.zeros(len(df))
    d_values = np.zeros(len(df))
    k_values[0] = 50
    d_values[0] = 50
    for i in range(1, len(df)):
        k_values[i] = 2 / 3 * k_values[i - 1] + 1 / 3 * rsv.iloc[i]
        d_values[i] = 2 / 3 * d_values[i - 1] + 1 / 3 * k_values[i]
    j_values = 3 * k_values - 2 * d_values
    df[f'K_{n}'] = k_values
    df[f'D_{n}'] = d_values
    df[f'J_{n}'] = j_values
    return df[[f'K_{n}', f'D_{n}', f'J_{n}']]

def calculate_ITS(df, n, column_name):
    required_columns = [f'K_{n}', f'D_{n}']
    for col in required_columns:
    Kt = df[f'K_{n}']
    Dt = df[f'D_{n}']
    ITS = pd.Series(0, index=df.index)
    K_previous = Kt.shift(1)
    buy_signal = (Kt <= Dt) & (Kt > K_previous)
    ITS[buy_signal] = 1
    sell_signal = (Kt >= Dt) & (Kt < K_previous)
    ITS[sell_signal] = -1
    df[column_name] = ITS
    return df[column_name]

def calculate_bias(df, n, column_name):
    MA = df['close'].rolling(window=n).mean()
    BIAS = (df['close'] - MA) / MA * 100
    df[column_name] = BIAS
    return df[column_name]

def calculate_psy(df, period, column_name):
    df['N_days_up'] = df['close'] > df['close'].shift(1)
    df['N_days_up'] = df['N_days_up'].astype('Int64')
    df['PSY'] = df['N_days_up'].rolling(window=period).mean() * 100
    df[column_name] = df['PSY']
    return df[column_name]

def calculate_dmi(df, n):
    df['HL'] = df['high'] - df['low']
    df['HC'] = abs(df['high'] - df['close'].shift(1))
    df['LC'] = abs(df['low'] - df['close'].shift(1))
    TR = df[['HL', 'HC', 'LC']].max(axis=1)
    df['+DM'] = np.where(df['high'] - df['high'].shift(1) > df['low'].shift(1) - df['low'], np.maximum(df['high'] - df['high'].shift(1), 0), 0)
    df['-DM'] = np.where(df['low'].shift(1) - df['low'] > df['high'] - df['high'].shift(1), np.maximum(df['low'].shift(1) - df['low'], 0), 0)
    df[f'+DI_{n}'] = 100 * (df['+DM'].rolling(window=n, min_periods=1).sum() / TR.rolling(window=n, min_periods=1).sum())
    df[f'-DI_{n}'] = 100 * (df['-DM'].rolling(window=n, min_periods=1).sum() / TR.rolling(window=n, min_periods=1).sum())
    df[f'DX_{n}'] = 100 * (abs(df[f'+DI_{n}'] - df[f'-DI_{n}']) / (df[f'+DI_{n}'] + df[f'-DI_{n}']))
    df[f'ADX_{n}'] = df[f'DX_{n}'].rolling(window=n, min_periods=1).mean()
    return df[[f'+DI_{n}', f'-DI_{n}', f'ADX_{n}']]

def calculate_cci(df, n, column_name):
    df['TP'] = (df['high'] + df['low'] + df['close']) / 3
    df['MA_TP'] = df['TP'].rolling(window=n).mean()
    df['MD'] = df['TP'].rolling(window=n).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    df[f'CCI_{n}'] = (df['TP'] - df['MA_TP']) / (0.015 * df['MD'].replace(0, 1e-10))
    df[column_name] = df[f'CCI_{n}']
    return df[column_name]

def calculate_mfi(df, n, column_name):
    required_columns = ['high', 'low', 'close', 'volume']
    for col in required_columns:
    tp = (df['high'] + df['low'] + df['close']) / 3
    rmf = tp * df['volume']
    positive_flow = np.where(tp > tp.shift(1), rmf, 0)
    negative_flow = np.where(tp < tp.shift(1), rmf, 0)
    positive_mf = pd.Series(positive_flow).rolling(window=n).sum()
    negative_mf = pd.Series(negative_flow).rolling(window=n).sum()
    mf_ratio = positive_mf / (negative_mf + 1e-10)
    df[f'MFI_{n}'] = 100 - 100 / (1 + mf_ratio)
    df[column_name] = df[f'MFI_{n}']
    return df[f'MFI_{n}']

def calculate_vr(df, n, column_name):
    df['Volume_Up'] = df['volume'].where(df['close'] > df['close'].shift(1), 0)
    df['Volume_Down'] = df['volume'].where(df['close'] < df['close'].shift(1), 0)
    df['Sum_Volume_Up'] = df['Volume_Up'].rolling(window=n).sum()
    df['Sum_Volume_Down'] = df['Volume_Down'].rolling(window=n).sum()
    df[column_name] = df['Sum_Volume_Up'] / (df['Sum_Volume_Down'] + 1e-10) * 100
    return df[column_name]
