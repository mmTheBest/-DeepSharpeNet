import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, List, Optional, Dict, Tuple

class SlidingWindowDataset(Dataset):

    def __init__(self, csv_path: Union[Path, str], win: int=30, target_shift: int=1, features: Optional[List[str]]=None, label_col: str='risk_adjusted_return', dist_label_col: str='distance_to_extremum', normalize: bool=True, stats: Optional[Dict]=None):
        self.df = pd.read_csv(csv_path)
        self.win = win
        self.shift = target_shift
        
        # Check if risk_adjusted_return exists in columns and use it by default
        if 'risk_adjusted_return' in self.df.columns:
            self.label_col = 'risk_adjusted_return'
        elif label_col in self.df.columns:
            self.label_col = label_col
        else:
            # Fall back to 'ret' if other targets don't exist
            self.label_col = 'ret'
        
        self.dist_label_col = dist_label_col
        if features is None:
            excluded_meta = ['date', 'code', 'preclose', 'amount', 'adjustflag', 'turn', 'tradestatus', 'pctchg', 'isst']
            excluded = set(excluded_meta + [self.label_col, self.dist_label_col] + [c for c in self.df.columns if c.startswith('action_order')])
            self.features = [c for c in self.df.columns if c not in excluded]
        else:
            self.features = features
        if normalize:
            if stats is None:
                self.stats = {c: (self.df[c].mean(), self.df[c].std() + 1e-08) for c in self.features}
            else:
                self.stats = stats
            for c in self.features:
                (μ, σ) = self.stats[c]
                self.df[c] = (self.df[c] - μ) / σ
        else:
            self.stats = None
        self.df = self.df.reset_index(drop=True)

    def __len__(self):
        return max(0, len(self.df) - self.win - self.shift)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if idx < 0 or idx >= len(self):
            raise IndexError(f'Index {idx} is out of bounds for dataset of length {len(self)}')
        start = idx
        end = idx + self.win
        window = self.df.loc[start:end - 1, self.features].values.astype(np.float32)
        target_idx = end + self.shift - 1
        if target_idx >= len(self.df):
            raise IndexError(f'Target index {target_idx} is out of bounds for dataframe of length {len(self.df)}')
        label = self.df.loc[target_idx, self.label_col].astype(np.float32)
        dist_label = self.df.loc[target_idx, self.dist_label_col].astype(np.float32)
        return (torch.tensor(window), torch.tensor(label), torch.tensor(dist_label))
if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('csv', help='path to *_basic.csv produced by data_pre_basic.py')
    p.add_argument('--win', type=int, default=30)
    args = p.parse_args()
    ds = SlidingWindowDataset(args.csv, win=args.win)
    print(f'Dataset length: {len(ds)} windows; one sample shape = {ds[0][0].shape}')
    print(f'Main label: {ds[0][1]}, Distance label: {ds[0][2]}')