import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

# 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # 注册为缓冲区，不作为模型参数
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        return x + self.pe[:, :x.size(1), :]

# ProbSparse自注意力机制
class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.dropout = nn.Dropout(attention_dropout)
        
    def _prob_QK(self, Q, K, sample_k, n_top):
        # Q: [B, H, L, D]
        B, H, L, E = K.shape
        _, _, S, _ = Q.shape
        
        # 计算样本数量
        sample_k = min(sample_k, L)
        
        # 随机采样K
        K_expand = K.unsqueeze(-3).expand(B, H, S, L, E)
        index_sample = torch.randint(0, L, (S, sample_k))
        K_sample = K_expand[:, :, torch.arange(S).unsqueeze(1), index_sample, :]
        
        # 计算Q和采样的K的点积
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)
        
        # 添加检查确保维度不为0
        if Q_K_sample.size(-1) == 0:
            # 如果维度为0，返回空的结果
            return torch.empty(B, H, S, 0, device=Q.device), torch.empty(B, H, S, 0, dtype=torch.long, device=Q.device)
        
        # 找到最大的n_top个注意力分数
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L)
        M_top = M.topk(min(n_top, M.size(-1)), sorted=False)[1]
        
        # 使用这些索引获取Q和K
        Q_reduce = Q[torch.arange(B)[:, None, None], 
                     torch.arange(H)[None, :, None], 
                     M_top, :]
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))
        
        return Q_K, M_top
    
    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        V_sum = V.mean(dim=-2)
        return V_sum.unsqueeze(-2).expand(B, H, L_Q, D).clone()
    
    def _update_context(self, context_in, V, scores, index):
        B, H, L_Q, D = context_in.shape
        
        attn = torch.zeros(B, H, L_Q, scores.shape[-1], device=V.device)
        attn[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = scores
        
        context_in = context_in.clone()
        context_in[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = \
            torch.matmul(attn[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :], V)
            
        return context_in
    
    def forward(self, queries, keys, values, attn_mask=None):
        B, L, H, D = queries.shape
        _, S, _, _ = keys.shape
        
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        
        U = self.factor * np.ceil(np.log(S)).astype('int').item() if S > 1 else 1
        u = self.factor * np.ceil(np.log(L)).astype('int').item() if L > 1 else 1
        
        scores_top, index = self._prob_QK(queries, keys, u, U)
        
        # 添加检查确保scores_top不为空
        if scores_top.size(-1) == 0:
            return torch.zeros(B, H, L, D, device=queries.device).transpose(1, 2)
        
        # 添加缩放因子
        scale = self.scale or 1./math.sqrt(D)
        scores_top = scores_top * scale
        
        # 应用掩码
        if self.mask_flag and attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).expand(B, H, L, S)
            scores_top = scores_top.masked_fill(attn_mask[torch.arange(B)[:, None, None], 
                                                       torch.arange(H)[None, :, None], 
                                                       index, :] == 0, -np.inf)
        
        # 应用softmax和dropout
        scores_top = self.dropout(F.softmax(scores_top, dim=-1))
        
        # 获取上下文
        context = self._get_initial_context(values, L)
        context = self._update_context(context, values, scores_top, index)
        
        return context.transpose(1, 2)

# 多头注意力机制
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.out_projection = nn.Linear(d_model, d_model)
        
        self.attention = ProbAttention()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, queries, keys, values, attn_mask=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        
        # 线性投影
        queries = self.query_projection(queries).view(B, L, self.n_heads, self.d_k)
        keys = self.key_projection(keys).view(B, S, self.n_heads, self.d_k)
        values = self.value_projection(values).view(B, S, self.n_heads, self.d_k)
        
        # 注意力计算
        out = self.attention(queries, keys, values, attn_mask)
        
        # 拼接多头结果
        out = out.reshape(B, L, self.d_model)
        
        # 最终线性投影
        out = self.out_projection(out)
        
        return out

# 前馈网络
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

# Encoder层
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, attn_mask=None):
        # 自注意力
        attn_output = self.self_attn(x, x, x, attn_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

# Decoder层
class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, self_attn_mask=None, cross_attn_mask=None):
        # 自注意力
        attn_output = self.self_attn(x, x, x, self_attn_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 交叉注意力
        attn_output = self.cross_attn(x, enc_output, enc_output, cross_attn_mask)
        x = self.norm2(x + self.dropout(attn_output))
        
        # 前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x

# Encoder
class Encoder(nn.Module):
    def __init__(self, input_dim, d_model, n_heads, d_ff, num_layers, dropout=0.1):
        super(Encoder, self).__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, attn_mask=None):
        # 输入投影
        x = self.input_projection(x)
        
        # 位置编码
        x = self.pos_encoding(x)
        
        # Dropout
        x = self.dropout(x)
        
        # Encoder层
        for layer in self.layers:
            x = layer(x, attn_mask)
            
        return x

# Decoder
class Decoder(nn.Module):
    def __init__(self, output_dim, d_model, n_heads, d_ff, num_layers, dropout=0.1):
        super(Decoder, self).__init__()
        self.output_projection = nn.Linear(d_model, d_model)  # 修改输出投影维度
        self.pos_encoding = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, self_attn_mask=None, cross_attn_mask=None):
        # 位置编码
        x = self.pos_encoding(x)
        
        # Dropout
        x = self.dropout(x)
        
        # Decoder层
        for layer in self.layers:
            x = layer(x, enc_output, self_attn_mask, cross_attn_mask)
            
        # 输出投影
        x = self.output_projection(x)
        
        return x

# Informer模型
class Informer(nn.Module):
    def __init__(self, input_dim, output_dim, d_model=512, n_heads=8, d_ff=2048, 
                 enc_layers=3, dec_layers=2, dropout=0.1):
        super(Informer, self).__init__()
        
        self.encoder = Encoder(input_dim, d_model, n_heads, d_ff, enc_layers, dropout)
        self.decoder = Decoder(output_dim, d_model, n_heads, d_ff, dec_layers, dropout)
        # 修改投影层，增加一个中间层
        self.intermediate = nn.Linear(d_model, d_model // 2)
        self.activation = nn.ReLU()
        self.projection = nn.Linear(d_model // 2, output_dim)
        
    def forward(self, src, tgt):
        # src: [batch_size, src_seq_len, input_dim]
        # tgt: [batch_size, tgt_seq_len, input_dim]
        
        # 编码器前向传播
        enc_output = self.encoder(src)
        
        # 解码器前向传播
        dec_output = self.decoder(tgt, enc_output)
        
        # 最终输出投影
        intermediate_output = self.intermediate(dec_output)
        intermediate_output = self.activation(intermediate_output)
        output = self.projection(intermediate_output)
        
        return output
