"""
PPG-Former: 面向生理信号的多尺度时频融合Transformer

创新点：
1. 生理周期感知位置编码 - 融入心跳周期先验知识
2. 多尺度时频融合注意力 - 同时捕获时域和频域特征
3. 压力感知门控机制 - 自适应特征选择
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PhysiologicalPositionalEncoding(nn.Module):
    """
    生理周期感知位置编码
    融入心跳周期先验知识（60-100 bpm）
    """
    def __init__(self, d_model, max_len=5000, heart_rate_range=(60, 100)):
        super().__init__()
        self.d_model = d_model
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        avg_period = 60 / ((heart_rate_range[0] + heart_rate_range[1]) / 2)
        cardiac_freq = 2 * math.pi / (avg_period * 20)
        cardiac_dim = d_model // 4
        cardiac_pe = torch.zeros(max_len, cardiac_dim)
        for i in range(cardiac_dim):
            cardiac_pe[:, i] = torch.sin(position.squeeze() * cardiac_freq * (i + 1))
        
        pe[:, :cardiac_dim] = pe[:, :cardiac_dim] + 0.5 * cardiac_pe
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class MultiScaleConvBlock(nn.Module):
    """多尺度卷积特征提取"""
    def __init__(self, d_model, scales=[1, 3, 5, 7]):
        super().__init__()
        self.scales = scales
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(d_model, d_model // len(scales), kernel_size=s, padding=s//2),
                nn.BatchNorm1d(d_model // len(scales)),
                nn.GELU()
            ) for s in scales
        ])
        self.fusion = nn.Conv1d(d_model, d_model, kernel_size=1)
        
    def forward(self, x):
        x_conv = x.transpose(1, 2)
        L = x_conv.size(2)
        
        multi_scale_feats = []
        for conv in self.convs:
            feat = conv(x_conv)
            feat = feat[:, :, :L]
            multi_scale_feats.append(feat)
        
        fused = torch.cat(multi_scale_feats, dim=1)
        output = self.fusion(fused)
        return output.transpose(1, 2)


class TimeFreqAttention(nn.Module):
    """
    时频融合注意力机制
    同时在时域和频域计算注意力
    """
    def __init__(self, d_model, n_heads=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        
        self.time_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.freq_linear = nn.Linear(d_model, d_model)
        
        self.time_weight = nn.Parameter(torch.tensor(0.7))
        self.freq_weight = nn.Parameter(torch.tensor(0.3))
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, L, D = x.shape
        
        time_out, _ = self.time_attn(x, x, x)
        
        x_freq = torch.fft.rfft(x, dim=1)
        freq_real = x_freq.real
        freq_imag = x_freq.imag
        
        freq_magnitude = torch.sqrt(freq_real**2 + freq_imag**2 + 1e-8)
        freq_feat = self.freq_linear(freq_magnitude)
        freq_out = F.interpolate(freq_feat.transpose(1, 2), size=L, mode='linear', align_corners=False).transpose(1, 2)
        
        t_weight = torch.sigmoid(self.time_weight)
        f_weight = torch.sigmoid(self.freq_weight)
        
        output = t_weight * time_out + f_weight * freq_out
        return self.norm(x + self.dropout(output))


class StressAwareGating(nn.Module):
    """
    压力感知门控机制
    自适应选择与压力相关的特征
    """
    def __init__(self, d_model, reduction=4):
        super().__init__()
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(d_model, d_model // reduction),
            nn.ReLU(),
            nn.Linear(d_model // reduction, d_model),
            nn.Sigmoid()
        )
        
        self.spatial_gate = nn.Sequential(
            nn.Conv1d(d_model, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x_t = x.transpose(1, 2)
        
        channel_weight = self.channel_gate(x_t).unsqueeze(-1)
        x_t = x_t * channel_weight
        
        spatial_weight = self.spatial_gate(x_t)
        x_t = x_t * spatial_weight
        
        return x_t.transpose(1, 2)


class PPGFormerBlock(nn.Module):
    """PPG-Former基本模块"""
    def __init__(self, d_model, n_heads, d_ff, scales=[1, 3, 5, 7], dropout=0.1):
        super().__init__()
        self.multi_scale_conv = MultiScaleConvBlock(d_model, scales)
        self.time_freq_attn = TimeFreqAttention(d_model, n_heads, dropout)
        self.stress_gate = StressAwareGating(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
    def forward(self, x):
        x = self.norm1(x + self.multi_scale_conv(x))
        x = self.time_freq_attn(x)
        x = self.norm2(x + self.stress_gate(x))
        x = self.norm3(x + self.ffn(x))
        return x


class PPGFormerEncoder(nn.Module):
    """
    PPG-Former编码器
    用于处理原始PPG信号
    """
    def __init__(self, input_dim, d_model, n_heads=8, d_ff=512, 
                 num_layers=3, scales=[1, 3, 5, 7], dropout=0.1):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoding = PhysiologicalPositionalEncoding(d_model)
        
        self.layers = nn.ModuleList([
            PPGFormerBlock(d_model, n_heads, d_ff, scales, dropout)
            for _ in range(num_layers)
        ])
        
        self.output_norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoding(x)
        
        for layer in self.layers:
            x = layer(x)
        
        return self.output_norm(x)


class PPGFormer(nn.Module):
    """
    PPG-Former: 独立使用时的完整模型
    可作为基准模型与其他模型对比
    """
    def __init__(self, input_dim=1, output_dim=1, d_model=128, n_heads=8, 
                 d_ff=512, num_layers=3, scales=[1, 3, 5, 7], dropout=0.1):
        super().__init__()
        
        self.encoder = PPGFormerEncoder(
            input_dim=input_dim,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            num_layers=num_layers,
            scales=scales,
            dropout=dropout
        )
        
        self.regressor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim)
        )
        
    def forward(self, src, tgt=None):
        x = self.encoder(src)
        x = x.mean(dim=1)
        output = self.regressor(x)
        return output.unsqueeze(1)