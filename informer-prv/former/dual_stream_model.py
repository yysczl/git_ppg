"""
Dual-Stream Network + PPG-Former-DualStream 完整模型

创新点：
1. 跨模态交互注意力 - 实现PPG和PRV信号之间的信息交互
2. 自适应权重融合 - 学习两种信号的最优融合权重
3. 多任务学习头 - 压力回归与情绪分类联合学习
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ppg_former_model import PPGFormerEncoder


class PRVEncoder(nn.Module):
    """
    PRV信号编码器
    处理脉率变异性特征
    """
    def __init__(self, input_dim, d_model, num_layers=2, dropout=0.1):
        super().__init__()
        
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.GELU()
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=4, 
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        x = self.input_proj(x)
        
        x_conv = self.temporal_conv(x.transpose(1, 2)).transpose(1, 2)
        x = x + x_conv
        
        x = self.transformer(x)
        
        return self.output_norm(x)


class CrossModalAttention(nn.Module):
    """
    跨模态交互注意力
    实现PPG和PRV特征之间的信息交互
    """
    def __init__(self, d_model, n_heads=8, dropout=0.1):
        super().__init__()
        
        self.ppg_to_prv_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.prv_to_ppg_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        
        self.ppg_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        self.prv_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        
        self.ppg_norm = nn.LayerNorm(d_model)
        self.prv_norm = nn.LayerNorm(d_model)
        
    def forward(self, ppg_feat, prv_feat):
        ppg_enhanced, _ = self.ppg_to_prv_attn(ppg_feat, prv_feat, prv_feat)
        prv_enhanced, _ = self.prv_to_ppg_attn(prv_feat, ppg_feat, ppg_feat)
        
        ppg_gate = self.ppg_gate(torch.cat([ppg_feat, ppg_enhanced], dim=-1))
        prv_gate = self.prv_gate(torch.cat([prv_feat, prv_enhanced], dim=-1))
        
        ppg_out = self.ppg_norm(ppg_feat + ppg_gate * ppg_enhanced)
        prv_out = self.prv_norm(prv_feat + prv_gate * prv_enhanced)
        
        return ppg_out, prv_out


class DualStreamFusion(nn.Module):
    """
    双流协同融合模块
    融合PPG-Former输出和PRV特征
    """
    def __init__(self, d_model, n_heads=8, num_layers=2, dropout=0.1):
        super().__init__()
        
        self.cross_modal_layers = nn.ModuleList([
            CrossModalAttention(d_model, n_heads, dropout)
            for _ in range(num_layers)
        ])
        
        self.adaptive_fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 2),
            nn.Softmax(dim=-1)
        )
        
        self.final_fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, ppg_feat, prv_feat):
        for cross_attn in self.cross_modal_layers:
            ppg_feat, prv_feat = cross_attn(ppg_feat, prv_feat)
        
        ppg_global = ppg_feat.mean(dim=1)
        prv_global = prv_feat.mean(dim=1)
        
        combined = torch.cat([ppg_global, prv_global], dim=-1)
        weights = self.adaptive_fusion(combined)
        
        weighted_feat = weights[:, 0:1] * ppg_global + weights[:, 1:2] * prv_global
        
        concat_feat = torch.cat([weighted_feat, ppg_global + prv_global], dim=-1)
        fused_feat = self.final_fusion(concat_feat)
        
        return fused_feat, ppg_feat, prv_feat


class MultiTaskHead(nn.Module):
    """
    多任务学习头
    同时进行压力回归和情绪分类
    """
    def __init__(self, d_model, num_emotions=5, dropout=0.1):
        super().__init__()
        
        self.shared_layer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.stress_head = nn.Sequential(
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 1)
        )
        
        self.emotion_head = nn.Sequential(
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, num_emotions)
        )
        
        self.uncertainty_stress = nn.Parameter(torch.tensor(0.0))
        self.uncertainty_emotion = nn.Parameter(torch.tensor(0.0))
        
    def forward(self, x):
        shared_feat = self.shared_layer(x)
        
        stress_pred = self.stress_head(shared_feat)
        emotion_pred = self.emotion_head(shared_feat)
        
        return stress_pred, emotion_pred
    
    def compute_loss(self, stress_pred, stress_target, emotion_pred=None, emotion_target=None):
        """多任务损失计算（带不确定性加权）"""
        stress_loss = F.mse_loss(stress_pred.squeeze(), stress_target.squeeze())
        
        if emotion_pred is not None and emotion_target is not None:
            emotion_loss = F.cross_entropy(emotion_pred, emotion_target)
            
            weighted_stress_loss = stress_loss * torch.exp(-self.uncertainty_stress) + self.uncertainty_stress
            weighted_emotion_loss = emotion_loss * torch.exp(-self.uncertainty_emotion) + self.uncertainty_emotion
            
            total_loss = weighted_stress_loss + weighted_emotion_loss
            return total_loss, stress_loss, emotion_loss
        else:
            return stress_loss, stress_loss, torch.tensor(0.0, device=stress_pred.device)


class PPGFormerDualStream(nn.Module):
    """
    PPG-Former-DualStream: 融合多尺度时频Transformer与双流协同的多任务心理压力预测模型
    
    完整创新点：
    1. PPG-Former: 生理周期感知位置编码 + 多尺度时频融合注意力 + 压力感知门控
    2. Dual-Stream: 跨模态交互注意力 + 自适应权重融合
    3. 多任务学习: 不确定性加权的压力回归与情绪分类联合学习
    """
    def __init__(self, ppg_input_dim=1, prv_input_dim=1, d_model=128, n_heads=8, 
                 d_ff=512, ppg_layers=3, prv_layers=2, fusion_layers=2,
                 num_emotions=5, scales=[1, 3, 5, 7], dropout=0.1):
        super().__init__()
        
        self.ppg_encoder = PPGFormerEncoder(
            input_dim=ppg_input_dim,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            num_layers=ppg_layers,
            scales=scales,
            dropout=dropout
        )
        
        self.prv_encoder = PRVEncoder(
            input_dim=prv_input_dim,
            d_model=d_model,
            num_layers=prv_layers,
            dropout=dropout
        )
        
        self.dual_stream_fusion = DualStreamFusion(
            d_model=d_model,
            n_heads=n_heads,
            num_layers=fusion_layers,
            dropout=dropout
        )
        
        self.task_head = MultiTaskHead(
            d_model=d_model,
            num_emotions=num_emotions,
            dropout=dropout
        )
        
    def forward(self, ppg_input, prv_input=None):
        """
        前向传播
        
        Args:
            ppg_input: PPG信号 [batch_size, ppg_seq_len, ppg_dim]
            prv_input: PRV信号 [batch_size, prv_seq_len, prv_dim]，如果为None则使用PPG
            
        Returns:
            stress_pred: 压力预测值 [batch_size, 1, 1]
            emotion_pred: 情绪分类logits [batch_size, num_emotions]
        """
        ppg_feat = self.ppg_encoder(ppg_input)
        
        if prv_input is None:
            prv_input = ppg_input
        prv_feat = self.prv_encoder(prv_input)
        
        fused_feat, ppg_enhanced, prv_enhanced = self.dual_stream_fusion(ppg_feat, prv_feat)
        
        stress_pred, emotion_pred = self.task_head(fused_feat)
        
        return stress_pred.unsqueeze(1), emotion_pred
    
    def compute_loss(self, ppg_input, prv_input, stress_target, emotion_target=None):
        """计算多任务损失"""
        stress_pred, emotion_pred = self.forward(ppg_input, prv_input)
        stress_pred = stress_pred.squeeze()
        
        total_loss, stress_loss, emotion_loss = self.task_head.compute_loss(
            stress_pred, stress_target, emotion_pred, emotion_target
        )
        return total_loss, stress_loss, emotion_loss


class DualStreamOnly(nn.Module):
    """
    Dual-Stream Only模型
    仅用于单模态数据（PPG或PRV），不使用PPG-Former
    可作为消融实验的对照模型
    """
    def __init__(self, input_dim=1, output_dim=1, d_model=128, n_heads=8,
                 num_layers=3, dropout=0.1):
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
        
    def forward(self, src, tgt=None):
        x = self.encoder(src)
        x = x.mean(dim=1)
        output = self.regressor(x)
        return output.unsqueeze(1)
        