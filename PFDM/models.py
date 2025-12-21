"""
PPG-Former-DualStream 模型定义
包含所有独立的模型组件，支持消融实验

模块结构:
1. 位置编码模块
   - StandardPositionalEncoding: 标准正弦位置编码
   - PhysiologicalPositionalEncoding: 生理周期感知位置编码
2. PPG-Former核心模块
   - MultiScaleConvBlock: 多尺度卷积特征提取
   - TimeFreqAttention: 时频融合注意力机制
   - StressAwareGating: 压力感知门控机制
   - PPGFormerBlock: PPG-Former基本模块
   - PPGFormerEncoder: PPG-Former编码器
3. PRV编码器
   - PRVEncoder: PRV信号编码器
4. 双流融合模块
   - CrossModalAttention: 跨模态交互注意力
   - DualStreamFusion: 双流协同融合模块
5. 多任务学习头
   - MultiTaskHead: 多任务学习头（压力回归+情绪分类）
6. 完整模型
   - PPGFormer: 单流PPG-Former模型
   - PPGFormerDualStream: 双流完整模型
   - DualStreamOnly: 仅PRV流模型（消融用）
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 第一部分：位置编码模块
# ============================================================

class StandardPositionalEncoding(nn.Module):
    """
    标准正弦位置编码
    基于 "Attention Is All You Need" (Vaswani et al., 2017)
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class PhysiologicalPositionalEncoding(nn.Module):
    """
    生理周期感知位置编码
    
    创新点：
    - 融入心跳周期先验知识（60-100 bpm）
    - 在标准位置编码基础上叠加心跳周期编码
    - 使模型在训练初期就具备对PPG信号周期性的感知能力
    
    理论基础：
    - 标准PE的基频（10000）与PPG的物理频率（1-2Hz心跳）无关
    - 将心率范围(60-100bpm)作为先验知识嵌入初始化空间
    """
    
    def __init__(self, d_model: int, max_len: int = 5000,
                 heart_rate_range: tuple = (60, 100), dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        
        # 1. 标准位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 2. 心跳周期编码
        # 假设采样率约为20Hz，心率60-100bpm对应周期0.6-1.0秒
        # 平均周期约0.75秒，对应约15个采样点
        avg_heart_rate = (heart_rate_range[0] + heart_rate_range[1]) / 2
        avg_period = 60 / avg_heart_rate  # 秒
        sampling_rate = 20  # 假设采样率20Hz
        cardiac_freq = 2 * math.pi / (avg_period * sampling_rate)
        
        # 在前1/4维度叠加心跳周期编码
        cardiac_dim = d_model // 4
        cardiac_pe = torch.zeros(max_len, cardiac_dim)
        
        for i in range(cardiac_dim):
            # 使用不同谐波频率
            harmonic = i + 1
            cardiac_pe[:, i] = torch.sin(position.squeeze() * cardiac_freq * harmonic)
        
        # 叠加心跳周期编码（权重可调）
        pe[:, :cardiac_dim] = pe[:, :cardiac_dim] + 0.5 * cardiac_pe
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
        # 可学习的权重参数
        self.alpha = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            [batch_size, seq_len, d_model]
        """
        x = x + self.alpha * self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ============================================================
# 第二部分：PPG-Former核心模块
# ============================================================

class MultiScaleConvBlock(nn.Module):
    """
    多尺度卷积特征提取模块
    
    设计原理：
    - 使用多种卷积核尺寸并行处理输入
    - 小卷积核(1x1, 3x3)捕获高频细节（如重搏波切迹）
    - 大卷积核(5x5, 7x7)捕获低频形态（如收缩/舒张期波形）
    - 拼接后融合多尺度特征
    
    类似于Inception模块的设计思想
    """
    
    def __init__(self, d_model: int, scales: list = [1, 3, 5, 7]):
        super().__init__()
        self.scales = scales
        self.num_scales = len(scales)
        
        # 每个尺度的输出维度
        scale_dim = d_model // self.num_scales
        
        # 多尺度卷积分支
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(d_model, scale_dim, kernel_size=s, padding=s // 2),
                nn.BatchNorm1d(scale_dim),
                nn.GELU()
            ) for s in scales
        ])
        
        # 1x1卷积融合
        self.fusion = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            [batch_size, seq_len, d_model]
        """
        # 转换为卷积格式: [B, C, L]
        x_conv = x.transpose(1, 2)
        L = x_conv.size(2)
        
        # 多尺度卷积
        multi_scale_feats = []
        for conv in self.convs:
            feat = conv(x_conv)
            # 确保输出长度一致
            feat = feat[:, :, :L]
            multi_scale_feats.append(feat)
        
        # 拼接多尺度特征
        fused = torch.cat(multi_scale_feats, dim=1)
        
        # 1x1卷积融合
        output = self.fusion(fused)
        
        # 转换回: [B, L, C]
        output = output.transpose(1, 2)
        
        return self.norm(output)


class TimeFreqAttention(nn.Module):
    """
    时频融合注意力机制
    
    创新点：
    - 同时在时域和频域计算注意力
    - 时域捕获局部模式和瞬态变化
    - 频域捕获周期性成分和全局特征
    - 自适应学习时频特征的融合权重
    
    对应开题报告中的"多尺度时频融合注意力机制(MSTF-Block)"
    """
    
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1,
                 use_freq_attention: bool = True):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.use_freq_attention = use_freq_attention
        
        # 时域注意力（标准多头自注意力）
        self.time_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        
        # 频域特征处理
        if use_freq_attention:
            self.freq_linear = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout)
            )
        
        # 自适应融合权重
        self.time_weight = nn.Parameter(torch.tensor(0.7))
        self.freq_weight = nn.Parameter(torch.tensor(0.3))
        
        # 层归一化和dropout
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            [batch_size, seq_len, d_model]
        """
        B, L, D = x.shape
        
        # 时域注意力
        time_out, _ = self.time_attn(x, x, x)
        
        if self.use_freq_attention:
            # 频域处理
            # FFT变换
            x_freq = torch.fft.rfft(x, dim=1)
            
            # 提取幅度谱（保留相位信息用于重建）
            freq_real = x_freq.real
            freq_imag = x_freq.imag
            freq_magnitude = torch.sqrt(freq_real ** 2 + freq_imag ** 2 + 1e-8)
            
            # 频域特征增强
            freq_feat = self.freq_linear(freq_magnitude)
            
            # 插值回原始时域长度
            freq_out = F.interpolate(
                freq_feat.transpose(1, 2),
                size=L,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)
            
            # 自适应权重融合
            t_weight = torch.sigmoid(self.time_weight)
            f_weight = torch.sigmoid(self.freq_weight)
            
            output = t_weight * time_out + f_weight * freq_out
        else:
            output = time_out
        
        # 残差连接和归一化
        return self.norm(x + self.dropout(output))


class StressAwareGating(nn.Module):
    """
    压力感知门控机制
    
    设计原理：
    - 通道门控：学习每个特征通道的重要性权重
    - 空间/时间门控：学习每个时间点的重要性权重
    - 双重门控机制自适应选择与压力相关的特征
    
    类似于SE-Block(通道)和CBAM(空间)的组合
    公式：X_out = X · σ(MLP(AvgPool(X))) · σ(Conv1D(X))
    """
    
    def __init__(self, d_model: int, reduction: int = 4):
        super().__init__()
        
        # 通道门控（类似SE-Block）
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(d_model, d_model // reduction),
            nn.ReLU(),
            nn.Linear(d_model // reduction, d_model),
            nn.Sigmoid()
        )
        
        # 空间/时间门控
        self.spatial_gate = nn.Sequential(
            nn.Conv1d(d_model, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            [batch_size, seq_len, d_model]
        """
        # 转换为: [B, C, L]
        x_t = x.transpose(1, 2)
        
        # 通道门控
        channel_weight = self.channel_gate(x_t).unsqueeze(-1)  # [B, C, 1]
        x_t = x_t * channel_weight
        
        # 空间门控
        spatial_weight = self.spatial_gate(x_t)  # [B, 1, L]
        x_t = x_t * spatial_weight
        
        # 转换回: [B, L, C]
        return x_t.transpose(1, 2)


class PPGFormerBlock(nn.Module):
    """
    PPG-Former基本模块
    
    结构：
    1. 多尺度卷积 + 残差
    2. 时频融合注意力
    3. 压力感知门控 + 残差
    4. 前馈网络 + 残差
    """
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 scales: list = [1, 3, 5, 7], dropout: float = 0.1,
                 use_multi_scale_conv: bool = True,
                 use_time_freq_attention: bool = True,
                 use_freq_attention: bool = True,
                 use_stress_gating: bool = True):
        super().__init__()
        
        self.use_multi_scale_conv = use_multi_scale_conv
        self.use_stress_gating = use_stress_gating
        
        # 多尺度卷积
        if use_multi_scale_conv:
            self.multi_scale_conv = MultiScaleConvBlock(d_model, scales)
        
        # 时频融合注意力
        self.time_freq_attn = TimeFreqAttention(
            d_model, n_heads, dropout, use_freq_attention
        ) if use_time_freq_attention else nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        
        # 压力感知门控
        if use_stress_gating:
            self.stress_gate = StressAwareGating(d_model)
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self._use_time_freq_attention = use_time_freq_attention
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            [batch_size, seq_len, d_model]
        """
        # 多尺度卷积
        if self.use_multi_scale_conv:
            x = self.norm1(x + self.multi_scale_conv(x))
        
        # 时频融合注意力
        if self._use_time_freq_attention:
            x = self.time_freq_attn(x)
        else:
            attn_out, _ = self.time_freq_attn(x, x, x)
            x = self.norm2(x + attn_out)
        
        # 压力感知门控
        if self.use_stress_gating:
            x = self.norm2(x + self.stress_gate(x))
        
        # 前馈网络
        x = self.norm3(x + self.ffn(x))
        
        return x


class PPGFormerEncoder(nn.Module):
    """
    PPG-Former编码器
    
    用于处理原始PPG信号，提取深层特征
    
    结构：
    1. 输入投影层
    2. 位置编码（可选生理周期感知）
    3. 多层PPGFormerBlock
    4. 输出归一化
    """
    
    def __init__(self, input_dim: int, d_model: int, n_heads: int = 8,
                 d_ff: int = 512, num_layers: int = 3,
                 scales: list = [1, 3, 5, 7], dropout: float = 0.1,
                 use_physiological_pe: bool = True,
                 use_multi_scale_conv: bool = True,
                 use_time_freq_attention: bool = True,
                 use_freq_attention: bool = True,
                 use_stress_gating: bool = True):
        super().__init__()
        
        # 输入投影
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # 位置编码
        if use_physiological_pe:
            self.pos_encoding = PhysiologicalPositionalEncoding(d_model, dropout=dropout)
        else:
            self.pos_encoding = StandardPositionalEncoding(d_model, dropout=dropout)
        
        # PPGFormer层
        self.layers = nn.ModuleList([
            PPGFormerBlock(
                d_model, n_heads, d_ff, scales, dropout,
                use_multi_scale_conv, use_time_freq_attention,
                use_freq_attention, use_stress_gating
            )
            for _ in range(num_layers)
        ])
        
        # 输出归一化
        self.output_norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, input_dim]
        Returns:
            [batch_size, seq_len, d_model]
        """
        # 输入投影
        x = self.input_proj(x)
        
        # 位置编码
        x = self.pos_encoding(x)
        
        # PPGFormer层
        for layer in self.layers:
            x = layer(x)
        
        return self.output_norm(x)


# ============================================================
# 第三部分：PRV编码器
# ============================================================

class PRVEncoder(nn.Module):
    """
    PRV信号编码器
    
    设计特点：
    - 轻量化设计（PRV序列较短，约80个点）
    - CNN + Transformer混合结构
    - 卷积提取局部时序模式
    - Transformer建模全局依赖
    """
    
    def __init__(self, input_dim: int, d_model: int,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        
        # 输入投影
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 时序卷积
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.GELU()
        )
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 输出归一化
        self.output_norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, input_dim]
        Returns:
            [batch_size, seq_len, d_model]
        """
        # 输入投影
        x = self.input_proj(x)
        
        # 时序卷积（残差连接）
        x_conv = self.temporal_conv(x.transpose(1, 2)).transpose(1, 2)
        x = x + x_conv
        
        # Transformer编码
        x = self.transformer(x)
        
        return self.output_norm(x)


# ============================================================
# 第四部分：双流融合模块
# ============================================================

class CrossModalAttention(nn.Module):
    """
    跨模态交互注意力
    
    创新点：
    - 双向交叉注意力实现PPG和PRV的信息交互
    - PPG→PRV: 利用PRV节律信息解释PPG波形变化
    - PRV→PPG: 利用PPG形态信息验证PRV特征可靠性
    - 门控机制控制融合程度，防止过度融合
    """
    
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        # PPG查询PRV
        self.ppg_to_prv_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        
        # PRV查询PPG
        self.prv_to_ppg_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        
        # 门控网络
        self.ppg_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        self.prv_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        
        # 层归一化
        self.ppg_norm = nn.LayerNorm(d_model)
        self.prv_norm = nn.LayerNorm(d_model)
    
    def forward(self, ppg_feat: torch.Tensor, prv_feat: torch.Tensor
                ) -> tuple:
        """
        Args:
            ppg_feat: [batch_size, ppg_seq_len, d_model]
            prv_feat: [batch_size, prv_seq_len, d_model]
        Returns:
            ppg_out: [batch_size, ppg_seq_len, d_model]
            prv_out: [batch_size, prv_seq_len, d_model]
        """
        # PPG增强（查询PRV）
        ppg_enhanced, _ = self.ppg_to_prv_attn(ppg_feat, prv_feat, prv_feat)
        
        # PRV增强（查询PPG）
        prv_enhanced, _ = self.prv_to_ppg_attn(prv_feat, ppg_feat, ppg_feat)
        
        # 门控融合
        ppg_gate = self.ppg_gate(torch.cat([ppg_feat, ppg_enhanced], dim=-1))
        prv_gate = self.prv_gate(torch.cat([prv_feat, prv_enhanced], dim=-1))
        
        # 残差连接
        ppg_out = self.ppg_norm(ppg_feat + ppg_gate * ppg_enhanced)
        prv_out = self.prv_norm(prv_feat + prv_gate * prv_enhanced)
        
        return ppg_out, prv_out


class DualStreamFusion(nn.Module):
    """
    双流协同融合模块
    
    功能：
    1. 多层跨模态交互注意力
    2. 自适应权重融合（根据信号质量动态调整）
    3. 最终特征融合
    """
    
    def __init__(self, d_model: int, n_heads: int = 8,
                 num_layers: int = 2, dropout: float = 0.1,
                 use_cross_modal_attention: bool = True):
        super().__init__()
        
        self.use_cross_modal_attention = use_cross_modal_attention
        
        # 跨模态交互层
        if use_cross_modal_attention:
            self.cross_modal_layers = nn.ModuleList([
                CrossModalAttention(d_model, n_heads, dropout)
                for _ in range(num_layers)
            ])
        
        # 自适应权重融合网络
        self.adaptive_fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 2),
            nn.Softmax(dim=-1)
        )
        
        # 最终融合层
        self.final_fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, ppg_feat: torch.Tensor, prv_feat: torch.Tensor
                ) -> tuple:
        """
        Args:
            ppg_feat: [batch_size, ppg_seq_len, d_model]
            prv_feat: [batch_size, prv_seq_len, d_model]
        Returns:
            fused_feat: [batch_size, d_model]
            ppg_feat: 增强后的PPG特征
            prv_feat: 增强后的PRV特征
        """
        # 跨模态交互
        if self.use_cross_modal_attention:
            for cross_attn in self.cross_modal_layers:
                ppg_feat, prv_feat = cross_attn(ppg_feat, prv_feat)
        
        # 全局平均池化
        ppg_global = ppg_feat.mean(dim=1)  # [B, d_model]
        prv_global = prv_feat.mean(dim=1)  # [B, d_model]
        
        # 计算自适应权重
        combined = torch.cat([ppg_global, prv_global], dim=-1)
        weights = self.adaptive_fusion(combined)  # [B, 2]
        
        # 加权融合
        weighted_feat = weights[:, 0:1] * ppg_global + weights[:, 1:2] * prv_global
        
        # 最终融合
        concat_feat = torch.cat([weighted_feat, ppg_global + prv_global], dim=-1)
        fused_feat = self.final_fusion(concat_feat)
        
        return fused_feat, ppg_feat, prv_feat


# ============================================================
# 第五部分：任务学习头
# ============================================================

class RegressionHead(nn.Module):
    """
    压力回归头
    
    用于单任务压力回归，将特征维度转换为输出维度
    """
    
    def __init__(self, d_model: int, output_dim: int = 1, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.output_dim = output_dim
        
        # 回归层：将d_model维度转换为output_dim
        self.regressor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, d_model]
        Returns:
            [batch_size, output_dim]
        """
        return self.regressor(x)
    
    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算MSE损失"""
        return F.mse_loss(pred.squeeze(), target.squeeze())


class ClassificationHead(nn.Module):
    """
    情绪分类头
    
    用于单任务情绪分类，将特征维度转换为类别数
    """
    
    def __init__(self, d_model: int, num_classes: int = 5, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_classes = num_classes
        
        # 分类层：将d_model维度转换为num_classes
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, d_model]
        Returns:
            [batch_size, num_classes]
        """
        return self.classifier(x)
    
    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算交叉熔损失"""
        return F.cross_entropy(pred, target)


class MultiTaskHead(nn.Module):
    """
    多任务学习头
    
    功能：
    1. 共享特征层
    2. 压力回归头
    3. 情绪分类头
    4. 不确定性加权损失计算
    
    基于 Kendall et al. (CVPR 2018) 的同方差不确定性理论
    """
    
    def __init__(self, d_model: int, num_emotions: int = 5, dropout: float = 0.1,
                 use_uncertainty_weighting: bool = True):
        super().__init__()
        
        self.use_uncertainty_weighting = use_uncertainty_weighting
        
        # 共享特征层
        self.shared_layer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 压力回归头
        self.stress_head = nn.Sequential(
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 1)
        )
        
        # 情绪分类头
        self.emotion_head = nn.Sequential(
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, num_emotions)
        )
        
        # 不确定性参数（可学习）
        if use_uncertainty_weighting:
            self.log_sigma_stress = nn.Parameter(torch.tensor(0.0))
            self.log_sigma_emotion = nn.Parameter(torch.tensor(0.0))
    
    def forward(self, x: torch.Tensor) -> tuple:
        """
        Args:
            x: [batch_size, d_model]
        Returns:
            stress_pred: [batch_size, 1]
            emotion_pred: [batch_size, num_emotions]
        """
        # 共享特征
        shared_feat = self.shared_layer(x)
        
        # 压力预测
        stress_pred = self.stress_head(shared_feat)
        
        # 情绪预测
        emotion_pred = self.emotion_head(shared_feat)
        
        return stress_pred, emotion_pred
    
    def compute_loss(self, stress_pred: torch.Tensor, stress_target: torch.Tensor,
                     emotion_pred: torch.Tensor = None, emotion_target: torch.Tensor = None
                     ) -> tuple:
        """
        计算多任务损失（带不确定性加权）
        
        损失公式：
        L = (1/2σ_stress²) * L_MSE + log(σ_stress) + (1/2σ_emotion²) * L_CE + log(σ_emotion)
        
        Args:
            stress_pred: 压力预测值
            stress_target: 压力真实值
            emotion_pred: 情绪预测值（可选）
            emotion_target: 情绪真实标签（可选）
        
        Returns:
            total_loss, stress_loss, emotion_loss
        """
        # 压力回归损失
        stress_loss = F.mse_loss(stress_pred.squeeze(), stress_target.squeeze())
        
        if emotion_pred is not None and emotion_target is not None:
            # 情绪分类损失
            emotion_loss = F.cross_entropy(emotion_pred, emotion_target)
            
            if self.use_uncertainty_weighting:
                # 不确定性加权
                # σ = exp(log_σ)，所以 1/σ² = exp(-2*log_σ)
                weighted_stress_loss = (
                    stress_loss * torch.exp(-2 * self.log_sigma_stress) +
                    self.log_sigma_stress
                )
                weighted_emotion_loss = (
                    emotion_loss * torch.exp(-2 * self.log_sigma_emotion) +
                    self.log_sigma_emotion
                )
                total_loss = weighted_stress_loss + weighted_emotion_loss
            else:
                # 简单加权
                total_loss = stress_loss + emotion_loss
            
            return total_loss, stress_loss, emotion_loss
        else:
            return stress_loss, stress_loss, torch.tensor(0.0, device=stress_pred.device)


# ============================================================
# 第六部分：完整模型
# ============================================================

class PPGFormer(nn.Module):
    """
    PPG-Former: 单流PPG信号模型
    
    支持回归和分类任务，可作为基准模型与其他模型对比
    """
    
    def __init__(self, input_dim: int = 1, output_dim: int = 1,
                 d_model: int = 128, n_heads: int = 8, d_ff: int = 512,
                 num_layers: int = 3, num_classes: int = 5,
                 scales: list = [1, 3, 5, 7], dropout: float = 0.1,
                 task_type: str = "regression",  # regression/classification/multi_task
                 use_physiological_pe: bool = True,
                 use_multi_scale_conv: bool = True,
                 use_time_freq_attention: bool = True,
                 use_freq_attention: bool = True,
                 use_stress_gating: bool = True,
                 use_uncertainty_weighting: bool = True):
        super().__init__()
        
        self.task_type = task_type
        
        self.encoder = PPGFormerEncoder(
            input_dim=input_dim,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            num_layers=num_layers,
            scales=scales,
            dropout=dropout,
            use_physiological_pe=use_physiological_pe,
            use_multi_scale_conv=use_multi_scale_conv,
            use_time_freq_attention=use_time_freq_attention,
            use_freq_attention=use_freq_attention,
            use_stress_gating=use_stress_gating
        )
        
        # 根据任务类型创建相应的头
        if task_type == "regression":
            self.task_head = RegressionHead(d_model, output_dim, dropout)
        elif task_type == "classification":
            self.task_head = ClassificationHead(d_model, num_classes, dropout)
        else:  # multi_task
            self.task_head = MultiTaskHead(
                d_model, num_classes, dropout, use_uncertainty_weighting
            )
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            src: [batch_size, seq_len, input_dim]
            tgt: 仅为兼容接口，不使用
        Returns:
            回归: [batch_size, 1]
            分类: [batch_size, num_classes]
            多任务: (stress_pred, emotion_pred)
        """
        # 编码
        x = self.encoder(src)
        
        # 全局平均池化
        x = x.mean(dim=1)
        
        # 任务头
        output = self.task_head(x)
        
        if self.task_type == "multi_task":
            return output  # (stress_pred, emotion_pred)
        elif self.task_type == "regression":
            return output.unsqueeze(1)  # [batch_size, 1, 1]
        else:  # classification
            return output  # [batch_size, num_classes]
    
    def compute_loss(self, src: torch.Tensor, stress_target: torch.Tensor = None,
                     emotion_target: torch.Tensor = None) -> tuple:
        """计算损失"""
        x = self.encoder(src)
        x = x.mean(dim=1)
        
        if self.task_type == "regression":
            pred = self.task_head(x)
            loss = self.task_head.compute_loss(pred, stress_target)
            return loss, loss, torch.tensor(0.0)
        elif self.task_type == "classification":
            pred = self.task_head(x)
            loss = self.task_head.compute_loss(pred, emotion_target)
            return loss, torch.tensor(0.0), loss
        else:  # multi_task
            stress_pred, emotion_pred = self.task_head(x)
            total_loss, stress_loss, emotion_loss = self.task_head.compute_loss(
                stress_pred, stress_target, emotion_pred, emotion_target
            )
            return total_loss, stress_loss, emotion_loss


class PPGFormerDualStream(nn.Module):
    """
    PPG-Former-DualStream: 融合多尺度时频Transformer与双流协同的多任务心理压力预测模型
    
    完整创新点：
    1. PPG-Former: 生理周期感知位置编码 + 多尺度时频融合注意力 + 压力感知门控
    2. Dual-Stream: 跨模态交互注意力 + 自适应权重融合
    3. 多任务学习: 不确定性加权的压力回归与情绪分类联合学习
    """
    
    def __init__(self, ppg_input_dim: int = 1, prv_input_dim: int = 1,
                 d_model: int = 128, n_heads: int = 8, d_ff: int = 512,
                 ppg_layers: int = 3, prv_layers: int = 2, fusion_layers: int = 2,
                 num_emotions: int = 5, scales: list = [1, 3, 5, 7],
                 dropout: float = 0.1,
                 use_physiological_pe: bool = True,
                 use_multi_scale_conv: bool = True,
                 use_time_freq_attention: bool = True,
                 use_freq_attention: bool = True,
                 use_stress_gating: bool = True,
                 use_cross_modal_attention: bool = True,
                 use_uncertainty_weighting: bool = True):
        super().__init__()
        
        # PPG编码器
        self.ppg_encoder = PPGFormerEncoder(
            input_dim=ppg_input_dim,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            num_layers=ppg_layers,
            scales=scales,
            dropout=dropout,
            use_physiological_pe=use_physiological_pe,
            use_multi_scale_conv=use_multi_scale_conv,
            use_time_freq_attention=use_time_freq_attention,
            use_freq_attention=use_freq_attention,
            use_stress_gating=use_stress_gating
        )
        
        # PRV编码器
        self.prv_encoder = PRVEncoder(
            input_dim=prv_input_dim,
            d_model=d_model,
            num_layers=prv_layers,
            dropout=dropout
        )
        
        # 双流融合
        self.dual_stream_fusion = DualStreamFusion(
            d_model=d_model,
            n_heads=n_heads,
            num_layers=fusion_layers,
            dropout=dropout,
            use_cross_modal_attention=use_cross_modal_attention
        )
        
        # 多任务头
        self.task_head = MultiTaskHead(
            d_model=d_model,
            num_emotions=num_emotions,
            dropout=dropout,
            use_uncertainty_weighting=use_uncertainty_weighting
        )
    
    def forward(self, ppg_input: torch.Tensor, prv_input: torch.Tensor = None
                ) -> tuple:
        """
        Args:
            ppg_input: [batch_size, ppg_seq_len, ppg_dim]
            prv_input: [batch_size, prv_seq_len, prv_dim]
        Returns:
            stress_pred: [batch_size, 1, 1]
            emotion_pred: [batch_size, num_emotions]
        """
        # PPG编码
        ppg_feat = self.ppg_encoder(ppg_input)
        
        # PRV编码
        if prv_input is None:
            prv_input = ppg_input
        prv_feat = self.prv_encoder(prv_input)
        
        # 双流融合
        fused_feat, ppg_enhanced, prv_enhanced = self.dual_stream_fusion(
            ppg_feat, prv_feat
        )
        
        # 多任务预测
        stress_pred, emotion_pred = self.task_head(fused_feat)
        
        return stress_pred.unsqueeze(1), emotion_pred
    
    def compute_loss(self, ppg_input: torch.Tensor, prv_input: torch.Tensor,
                     stress_target: torch.Tensor, emotion_target: torch.Tensor = None
                     ) -> tuple:
        """计算多任务损失"""
        stress_pred, emotion_pred = self.forward(ppg_input, prv_input)
        stress_pred = stress_pred.squeeze()
        
        total_loss, stress_loss, emotion_loss = self.task_head.compute_loss(
            stress_pred, stress_target, emotion_pred, emotion_target
        )
        
        return total_loss, stress_loss, emotion_loss


class PRVModel(nn.Module):
    """
    PRV独立模型
    
    支持回归和分类任务，用于单独使用PRV进行训练
    """
    
    def __init__(self, input_dim: int = 1, output_dim: int = 1,
                 d_model: int = 128, n_heads: int = 8,
                 num_layers: int = 2, num_classes: int = 5,
                 dropout: float = 0.1, task_type: str = "regression",
                 use_uncertainty_weighting: bool = True):
        super().__init__()
        
        self.task_type = task_type
        
        self.encoder = PRVEncoder(
            input_dim=input_dim,
            d_model=d_model,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # 根据任务类型创建相应的头
        if task_type == "regression":
            self.task_head = RegressionHead(d_model, output_dim, dropout)
        elif task_type == "classification":
            self.task_head = ClassificationHead(d_model, num_classes, dropout)
        else:  # multi_task
            self.task_head = MultiTaskHead(
                d_model, num_classes, dropout, use_uncertainty_weighting
            )
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            src: [batch_size, seq_len, input_dim]
            tgt: 仅为兼容接口，不使用
        Returns:
            回归: [batch_size, 1]
            分类: [batch_size, num_classes]
            多任务: (stress_pred, emotion_pred)
        """
        x = self.encoder(src)
        x = x.mean(dim=1)
        output = self.task_head(x)
        
        if self.task_type == "multi_task":
            return output
        elif self.task_type == "regression":
            return output.unsqueeze(1)
        else:
            return output
    
    def compute_loss(self, src: torch.Tensor, stress_target: torch.Tensor = None,
                     emotion_target: torch.Tensor = None) -> tuple:
        """计算损失"""
        x = self.encoder(src)
        x = x.mean(dim=1)
        
        if self.task_type == "regression":
            pred = self.task_head(x)
            loss = self.task_head.compute_loss(pred, stress_target)
            return loss, loss, torch.tensor(0.0)
        elif self.task_type == "classification":
            pred = self.task_head(x)
            loss = self.task_head.compute_loss(pred, emotion_target)
            return loss, torch.tensor(0.0), loss
        else:  # multi_task
            stress_pred, emotion_pred = self.task_head(x)
            total_loss, stress_loss, emotion_loss = self.task_head.compute_loss(
                stress_pred, stress_target, emotion_pred, emotion_target
            )
            return total_loss, stress_loss, emotion_loss


class DualStreamOnly(nn.Module):
    """
    Dual-Stream Only模型
    
    仅使用PRV流，不使用PPG-Former
    可作为消融实验的对照模型
    """
    
    def __init__(self, input_dim: int = 1, output_dim: int = 1,
                 d_model: int = 128, n_heads: int = 8,
                 num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        
        self.encoder = PRVEncoder(
            input_dim=input_dim,
            d_model=d_model,
            num_layers=num_layers,
            dropout=dropout
        )
        
        self.regressor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim)
        )
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            src: [batch_size, seq_len, input_dim]
            tgt: 仅为兼容接口，不使用
        Returns:
            [batch_size, 1, output_dim]
        """
        x = self.encoder(src)
        x = x.mean(dim=1)
        output = self.regressor(x)
        return output.unsqueeze(1)


# ============================================================
# 基线模型（用于对比实验）
# ============================================================

class LSTMBaseline(nn.Module):
    """LSTM基线模型"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 num_layers: int = 2, output_dim: int = 1):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor = None) -> torch.Tensor:
        out, _ = self.lstm(src)
        out = self.fc(out[:, -1, :])
        return out.unsqueeze(1)


class TransformerBaseline(nn.Module):
    """Transformer基线模型"""
    
    def __init__(self, input_dim: int, d_model: int = 128,
                 n_heads: int = 8, num_layers: int = 3,
                 output_dim: int = 1, dropout: float = 0.1):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4, dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.fc = nn.Linear(d_model, output_dim)
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor = None) -> torch.Tensor:
        x = self.input_proj(src)
        x = self.encoder(x)
        x = x.mean(dim=1)
        out = self.fc(x)
        return out.unsqueeze(1)


# ============================================================
# 模型工厂函数
# ============================================================

def create_model(model_name: str, config=None, **kwargs):
    """
    根据名称创建模型
    
    Args:
        model_name: 模型名称
        config: 配置对象
        **kwargs: 额外参数
    
    Returns:
        模型实例
    """
    models = {
        'ppg_former': PPGFormer,
        'prv_model': PRVModel,
        'ppg_former_dual_stream': PPGFormerDualStream,
        'dual_stream_only': DualStreamOnly,
        'lstm': LSTMBaseline,
        'transformer': TransformerBaseline,
    }
    
    if model_name.lower() not in models:
        raise ValueError(f"未知模型: {model_name}. 可用模型: {list(models.keys())}")
    
    model_class = models[model_name.lower()]
    
    if config is not None:
        # 获取任务类型
        task_type = getattr(config.training, 'task_type', 'regression')
        
        # 从配置创建模型
        if model_name.lower() == 'ppg_former_dual_stream':
            return model_class(
                ppg_input_dim=config.model.ppg_input_dim,
                prv_input_dim=config.model.prv_input_dim,
                d_model=config.model.d_model,
                n_heads=config.model.n_heads,
                d_ff=config.model.d_ff,
                ppg_layers=config.model.ppg_layers,
                prv_layers=config.model.prv_layers,
                fusion_layers=config.model.fusion_layers,
                num_emotions=config.model.num_emotions,
                scales=config.model.scales,
                dropout=config.model.dropout,
                use_physiological_pe=config.ablation.use_physiological_pe,
                use_multi_scale_conv=config.ablation.use_multi_scale_conv,
                use_time_freq_attention=config.ablation.use_time_freq_attention,
                use_freq_attention=config.ablation.use_freq_attention,
                use_stress_gating=config.ablation.use_stress_gating,
                use_cross_modal_attention=config.ablation.use_cross_modal_attention,
                use_uncertainty_weighting=config.ablation.use_uncertainty_weighting
            )
        elif model_name.lower() == 'ppg_former':
            return model_class(
                input_dim=config.model.ppg_input_dim,
                d_model=config.model.d_model,
                n_heads=config.model.n_heads,
                d_ff=config.model.d_ff,
                num_layers=config.model.ppg_layers,
                num_classes=config.model.num_emotions,
                scales=config.model.scales,
                dropout=config.model.dropout,
                task_type=task_type,
                use_physiological_pe=config.ablation.use_physiological_pe,
                use_multi_scale_conv=config.ablation.use_multi_scale_conv,
                use_time_freq_attention=config.ablation.use_time_freq_attention,
                use_freq_attention=config.ablation.use_freq_attention,
                use_stress_gating=config.ablation.use_stress_gating,
                use_uncertainty_weighting=config.ablation.use_uncertainty_weighting
            )
        elif model_name.lower() == 'prv_model':
            return model_class(
                input_dim=config.model.prv_input_dim,
                d_model=config.model.d_model,
                n_heads=config.model.n_heads,
                num_layers=config.model.prv_layers,
                num_classes=config.model.num_emotions,
                dropout=config.model.dropout,
                task_type=task_type,
                use_uncertainty_weighting=config.ablation.use_uncertainty_weighting
            )
    
    return model_class(**kwargs)


def get_model_for_train_mode(config) -> tuple:
    """
    根据训练模式获取相应的模型
    
    Args:
        config: 实验配置
    
    Returns:
        (model_class, model_name)
    """
    train_mode = config.training.train_mode
    task_type = config.training.task_type
    
    mode_model_map = {
        'ppg_only': ('ppg_former', 'PPGFormer'),
        'prv_only': ('prv_model', 'PRVModel'),
        'ppg_regression': ('ppg_former', 'PPGFormer'),
        'prv_regression': ('prv_model', 'PRVModel'),
        'ppg_classification': ('ppg_former', 'PPGFormer'),
        'prv_classification': ('prv_model', 'PRVModel'),
        'dual_stream': ('ppg_former_dual_stream', 'PPGFormerDualStream'),
        'multi_task': ('ppg_former_dual_stream', 'PPGFormerDualStream'),
    }
    
    if train_mode not in mode_model_map:
        raise ValueError(f"未知训练模式: {train_mode}")
    
    model_key, model_name = mode_model_map[train_mode]
    model = create_model(model_key, config)
    
    return model, model_name


# ============================================================
# 第七部分：基准模型包装器（支持回归/分类任务）
# ============================================================

class Chomp1d(nn.Module):
    """用于TCN的序列裁剪模块"""
    def __init__(self, padding):
        super(Chomp1d, self).__init__()
        self.padding = padding

    def forward(self, x):
        return x[:, :, :-self.padding] if self.padding > 0 else x


class TemporalBlock(nn.Module):
    """TCN的时序块"""
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1, 
                                self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class BaselineModelWrapper(nn.Module):
    """
    基准模型通用包装器
    
    支持为基准模型添加回归或分类头，使其支持四种单任务训练模式
    """
    
    def __init__(self, backbone: nn.Module, hidden_dim: int, 
                 task_type: str = 'regression', num_classes: int = 5,
                 dropout: float = 0.1):
        super().__init__()
        self.backbone = backbone
        self.task_type = task_type
        self.hidden_dim = hidden_dim
        
        # 根据任务类型创建头
        if task_type == 'regression':
            self.task_head = RegressionHead(hidden_dim, 1, dropout)
        elif task_type == 'classification':
            self.task_head = ClassificationHead(hidden_dim, num_classes, dropout)
        else:
            raise ValueError(f"未知任务类型: {task_type}")
    
    def forward(self, src, tgt=None):
        """
        Args:
            src: [batch_size, seq_len, input_dim]
        Returns:
            回归: [batch_size, 1]
            分类: [batch_size, num_classes]
        """
        # 使用backbone提取特征
        out = self.backbone(src, tgt)  # [batch_size, 1, hidden_dim] 或 [batch_size, 1, output_dim]
        
        # 如果backbone输出是三维的，取最后一个时间步
        if out.dim() == 3:
            out = out.squeeze(1)  # [batch_size, hidden_dim]
        
        # 通过任务头
        output = self.task_head(out)
        
        if self.task_type == 'regression':
            return output.unsqueeze(1)  # [batch_size, 1, 1]
        else:
            return output  # [batch_size, num_classes]
    
    def compute_loss(self, src, stress_target=None, emotion_target=None):
        """计算损失"""
        output = self.forward(src)
        
        if self.task_type == 'regression':
            loss = self.task_head.compute_loss(output.squeeze(), stress_target)
            return loss, loss, torch.tensor(0.0)
        else:
            loss = self.task_head.compute_loss(output, emotion_target)
            return loss, torch.tensor(0.0), loss


class LSTMModel(nn.Module):
    """
    LSTM基准模型，支持回归和分类任务
    """
    def __init__(self, input_dim: int = 1, hidden_dim: int = 128,
                 num_layers: int = 2, num_classes: int = 5,
                 dropout: float = 0.1, task_type: str = 'regression'):
        super().__init__()
        self.task_type = task_type
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        if task_type == 'regression':
            self.task_head = RegressionHead(hidden_dim, 1, dropout)
        else:
            self.task_head = ClassificationHead(hidden_dim, num_classes, dropout)
    
    def forward(self, src, tgt=None):
        h0 = torch.zeros(self.num_layers, src.size(0), self.hidden_dim).to(src.device)
        c0 = torch.zeros(self.num_layers, src.size(0), self.hidden_dim).to(src.device)
        
        out, _ = self.lstm(src, (h0, c0))
        out = out[:, -1, :]  # 取最后一个时间步
        
        output = self.task_head(out)
        
        if self.task_type == 'regression':
            return output.unsqueeze(1)
        return output
    
    def compute_loss(self, src, stress_target=None, emotion_target=None):
        output = self.forward(src)
        if self.task_type == 'regression':
            loss = self.task_head.compute_loss(output.squeeze(), stress_target)
            return loss, loss, torch.tensor(0.0)
        else:
            loss = self.task_head.compute_loss(output, emotion_target)
            return loss, torch.tensor(0.0), loss


class GRUModel(nn.Module):
    """
    GRU基准模型，支持回归和分类任务
    """
    def __init__(self, input_dim: int = 1, hidden_dim: int = 128,
                 num_layers: int = 2, num_classes: int = 5,
                 dropout: float = 0.1, task_type: str = 'regression'):
        super().__init__()
        self.task_type = task_type
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers,
                         batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        if task_type == 'regression':
            self.task_head = RegressionHead(hidden_dim, 1, dropout)
        else:
            self.task_head = ClassificationHead(hidden_dim, num_classes, dropout)
    
    def forward(self, src, tgt=None):
        h0 = torch.zeros(self.num_layers, src.size(0), self.hidden_dim).to(src.device)
        
        out, _ = self.gru(src, h0)
        out = out[:, -1, :]
        
        output = self.task_head(out)
        
        if self.task_type == 'regression':
            return output.unsqueeze(1)
        return output
    
    def compute_loss(self, src, stress_target=None, emotion_target=None):
        output = self.forward(src)
        if self.task_type == 'regression':
            loss = self.task_head.compute_loss(output.squeeze(), stress_target)
            return loss, loss, torch.tensor(0.0)
        else:
            loss = self.task_head.compute_loss(output, emotion_target)
            return loss, torch.tensor(0.0), loss


class BiLSTMModel(nn.Module):
    """
    双向LSTM基准模型，支持回归和分类任务
    """
    def __init__(self, input_dim: int = 1, hidden_dim: int = 128,
                 num_layers: int = 2, num_classes: int = 5,
                 dropout: float = 0.1, task_type: str = 'regression'):
        super().__init__()
        self.task_type = task_type
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                           batch_first=True, bidirectional=True,
                           dropout=dropout if num_layers > 1 else 0)
        
        # 双向，所以输出维度是hidden_dim * 2
        if task_type == 'regression':
            self.task_head = RegressionHead(hidden_dim * 2, 1, dropout)
        else:
            self.task_head = ClassificationHead(hidden_dim * 2, num_classes, dropout)
    
    def forward(self, src, tgt=None):
        h0 = torch.zeros(self.num_layers * 2, src.size(0), self.hidden_dim).to(src.device)
        c0 = torch.zeros(self.num_layers * 2, src.size(0), self.hidden_dim).to(src.device)
        
        out, _ = self.lstm(src, (h0, c0))
        out = out[:, -1, :]
        
        output = self.task_head(out)
        
        if self.task_type == 'regression':
            return output.unsqueeze(1)
        return output
    
    def compute_loss(self, src, stress_target=None, emotion_target=None):
        output = self.forward(src)
        if self.task_type == 'regression':
            loss = self.task_head.compute_loss(output.squeeze(), stress_target)
            return loss, loss, torch.tensor(0.0)
        else:
            loss = self.task_head.compute_loss(output, emotion_target)
            return loss, torch.tensor(0.0), loss


class TCNModel(nn.Module):
    """
    TCN（时序卷积网络）基准模型，支持回归和分类任务
    """
    def __init__(self, input_dim: int = 1, hidden_dim: int = 128,
                 num_layers: int = 4, num_classes: int = 5,
                 kernel_size: int = 3, dropout: float = 0.2,
                 task_type: str = 'regression'):
        super().__init__()
        self.task_type = task_type
        
        # 创建通道列表
        num_channels = [hidden_dim] * num_layers
        
        layers = []
        for i in range(num_layers):
            dilation_size = 2 ** i
            in_channels = input_dim if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers.append(TemporalBlock(
                in_channels, out_channels, kernel_size,
                stride=1, dilation=dilation_size,
                padding=(kernel_size-1) * dilation_size,
                dropout=dropout
            ))
        
        self.network = nn.Sequential(*layers)
        
        if task_type == 'regression':
            self.task_head = RegressionHead(hidden_dim, 1, dropout)
        else:
            self.task_head = ClassificationHead(hidden_dim, num_classes, dropout)
    
    def forward(self, src, tgt=None):
        # src: [batch_size, seq_len, input_dim]
        src = src.transpose(1, 2)  # [batch_size, input_dim, seq_len]
        out = self.network(src)
        out = out[:, :, -1]  # 取最后一个时间步
        
        output = self.task_head(out)
        
        if self.task_type == 'regression':
            return output.unsqueeze(1)
        return output
    
    def compute_loss(self, src, stress_target=None, emotion_target=None):
        output = self.forward(src)
        if self.task_type == 'regression':
            loss = self.task_head.compute_loss(output.squeeze(), stress_target)
            return loss, loss, torch.tensor(0.0)
        else:
            loss = self.task_head.compute_loss(output, emotion_target)
            return loss, torch.tensor(0.0), loss


class TransformerModel(nn.Module):
    """
    Transformer基准模型，支持回归和分类任务
    """
    def __init__(self, input_dim: int = 1, hidden_dim: int = 128,
                 n_heads: int = 8, num_layers: int = 3,
                 num_classes: int = 5, dropout: float = 0.1,
                 task_type: str = 'regression'):
        super().__init__()
        self.task_type = task_type
        self.hidden_dim = hidden_dim
        
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        if task_type == 'regression':
            self.task_head = RegressionHead(hidden_dim, 1, dropout)
        else:
            self.task_head = ClassificationHead(hidden_dim, num_classes, dropout)
    
    def forward(self, src, tgt=None):
        # 输入投影
        x = self.input_projection(src)
        
        # Transformer编码
        x = self.encoder(x)
        
        # 取最后一个时间步或平均池化
        x = x.mean(dim=1)
        
        output = self.task_head(x)
        
        if self.task_type == 'regression':
            return output.unsqueeze(1)
        return output
    
    def compute_loss(self, src, stress_target=None, emotion_target=None):
        output = self.forward(src)
        if self.task_type == 'regression':
            loss = self.task_head.compute_loss(output.squeeze(), stress_target)
            return loss, loss, torch.tensor(0.0)
        else:
            loss = self.task_head.compute_loss(output, emotion_target)
            return loss, torch.tensor(0.0), loss


class InformerModel(nn.Module):
    """
    Informer基准模型，支持回归和分类任务
    """
    def __init__(self, input_dim: int = 1, hidden_dim: int = 128,
                 n_heads: int = 8, num_layers: int = 3,
                 num_classes: int = 5, dropout: float = 0.1,
                 task_type: str = 'regression'):
        super().__init__()
        self.task_type = task_type
        self.hidden_dim = hidden_dim
        
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Informer风格的中间层
        self.intermediate = nn.Linear(hidden_dim, hidden_dim // 2)
        self.activation = nn.ReLU()
        
        if task_type == 'regression':
            self.task_head = RegressionHead(hidden_dim // 2, 1, dropout)
        else:
            self.task_head = ClassificationHead(hidden_dim // 2, num_classes, dropout)
    
    def forward(self, src, tgt=None):
        # 输入投影
        x = self.input_projection(src) * math.sqrt(self.hidden_dim)
        
        # Transformer编码
        x = self.encoder(x)
        
        # 取最后一个时间步
        x = x[:, -1, :]
        
        # 中间层
        x = self.intermediate(x)
        x = self.activation(x)
        
        output = self.task_head(x)
        
        if self.task_type == 'regression':
            return output.unsqueeze(1)
        return output
    
    def compute_loss(self, src, stress_target=None, emotion_target=None):
        output = self.forward(src)
        if self.task_type == 'regression':
            loss = self.task_head.compute_loss(output.squeeze(), stress_target)
            return loss, loss, torch.tensor(0.0)
        else:
            loss = self.task_head.compute_loss(output, emotion_target)
            return loss, torch.tensor(0.0), loss


# 基准模型工厂函数
BASELINE_MODELS = {
    'lstm': LSTMModel,
    'gru': GRUModel,
    'bilstm': BiLSTMModel,
    'tcn': TCNModel,
    'transformer_baseline': TransformerModel,
    'informer': InformerModel,
}


def create_baseline_model(model_name: str, input_dim: int = 1, hidden_dim: int = 128,
                          num_layers: int = 2, num_classes: int = 5,
                          dropout: float = 0.1, task_type: str = 'regression',
                          **kwargs):
    """
    创建基准模型
    
    Args:
        model_name: 模型名称 (lstm/gru/bilstm/tcn/transformer_baseline/informer)
        input_dim: 输入维度
        hidden_dim: 隐藏层维度
        num_layers: 层数
        num_classes: 分类类别数
        dropout: dropout比例
        task_type: 任务类型 (regression/classification)
    
    Returns:
        模型实例
    """
    if model_name.lower() not in BASELINE_MODELS:
        raise ValueError(f"未知基准模型: {model_name}. 可用模型: {list(BASELINE_MODELS.keys())}")
    
    model_class = BASELINE_MODELS[model_name.lower()]
    
    # 根据模型类型传递不同的参数
    if model_name.lower() in ['lstm', 'gru', 'bilstm']:
        return model_class(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout,
            task_type=task_type
        )
    elif model_name.lower() == 'tcn':
        return model_class(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout,
            task_type=task_type,
            kernel_size=kwargs.get('kernel_size', 3)
        )
    else:  # transformer_baseline, informer
        return model_class(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_heads=kwargs.get('n_heads', 8),
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout,
            task_type=task_type
        )


def list_available_models() -> list:
    """列出所有可用模型"""
    ppg_former_models = ['ppg_former', 'prv_model', 'ppg_former_dual_stream', 'dual_stream_only']
    baseline_models = list(BASELINE_MODELS.keys())
    return ppg_former_models + baseline_models
