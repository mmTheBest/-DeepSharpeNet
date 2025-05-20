from __future__ import annotations
from typing import Literal, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class DualAttention(nn.Module):

    def __init__(self, channels: int, reduction: int=8):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(channels, channels // reduction, bias=False), nn.ReLU(inplace=True), nn.Linear(channels // reduction, channels, bias=False), nn.Sigmoid())
        self.spatial = nn.Sequential(nn.Conv1d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ca = x.mean(dim=2)
        ca = self.mlp(ca).unsqueeze(-1)
        x_ca = x * ca
        sa = self.spatial(x)
        out = x_ca * sa
        return out

class GatedConv1d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super(GatedConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.gate_conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        conv_out = self.conv(x)
        gate = self.sigmoid(self.gate_conv(x))
        return conv_out * gate

class OutputScaler(nn.Module):

    def __init__(self, input_dim=1):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(input_dim))
        self.bias = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x):
        scaled = x * self.scale + self.bias
        return torch.tanh(scaled)

class ResidualBlock(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(), nn.Linear(dim, dim), nn.BatchNorm1d(dim))
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.layers(x))

class TemporalAttention(nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.Tanh(), nn.Linear(hidden_dim // 2, 1), nn.Softmax(dim=1))

    def forward(self, x):
        attention_weights = self.attention(x)
        return torch.sum(x * attention_weights, dim=1)

class FeatureImportance(nn.Module):

    def __init__(self, num_features):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(num_features))
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        weights = self.softmax(self.weights)
        return x * weights.unsqueeze(0).unsqueeze(0)

class CNNAttnLSTM(nn.Module):

    def __init__(self, in_channels: int, conv_channels: tuple[int, int]=(64, 128), lstm_hidden: int=128, output_dim: int=1, task: Literal['reg', 'cls']='reg'):
        super().__init__()
        (c1, c2) = conv_channels
        self.feature_importance = FeatureImportance(in_channels)
        self.conv1 = nn.Sequential(GatedConv1d(in_channels, c1, kernel_size=3, padding=1), nn.BatchNorm1d(c1), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(GatedConv1d(c1, c2, kernel_size=3, padding=1), nn.BatchNorm1d(c2), nn.ReLU(inplace=True))
        self.attn = DualAttention(c2)
        self.temp_attn = TemporalAttention(c2)
        self.lstm = nn.LSTM(input_size=c2, hidden_size=lstm_hidden, num_layers=2, batch_first=True, dropout=0.2, bidirectional=True)
        self.combine_bidirectional = nn.Linear(lstm_hidden * 2, lstm_hidden)
        self.bn = nn.BatchNorm1d(lstm_hidden)
        self.main_features = nn.Sequential(ResidualBlock(lstm_hidden), ResidualBlock(lstm_hidden), nn.Dropout(0.3))
        self.head = nn.Sequential(nn.Linear(lstm_hidden, lstm_hidden), nn.ReLU(), nn.Linear(lstm_hidden, output_dim), OutputScaler(output_dim))
        self.task = task

    def forward(self, x: torch.Tensor):
        batch_size = x.size(0)
        x = self.feature_importance(x)
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attn(x)
        x = x.permute(0, 2, 1)
        x = self.temp_attn(x)
        x = x.unsqueeze(1).expand(-1, 30, -1)
        (lstm_out, _) = self.lstm(x)
        lstm_out = self.combine_bidirectional(lstm_out)
        h = lstm_out[:, -1, :]
        h = self.bn(h)
        main_features = self.main_features(h)
        main_out = self.head(main_features)
        
        if self.task == 'cls':
            return main_out
        return main_out.squeeze(-1)

def build_model(sample_window: torch.Tensor, task: Literal['reg', 'cls']='reg', output_dim: int | None=None) -> CNNAttnLSTM:
    (_, seq_len, feat) = sample_window.shape
    if output_dim is None:
        output_dim = 1 if task == 'reg' else 3
    model = CNNAttnLSTM(in_channels=feat, output_dim=output_dim, task=task)
    return model
if __name__ == '__main__':
    import torch
    (batch, seq, feat) = (8, 30, 9)
    sample = torch.randn(batch, seq, feat)
    m = build_model(sample, task='reg')
    output = m(sample)
    print('output shape:', output.shape)
    
    # Test classification mode
    m_cls = build_model(sample, task='cls', output_dim=3)
    cls_output = m_cls(sample)
    print('classification output shape:', cls_output.shape)