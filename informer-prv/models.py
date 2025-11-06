import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


# LSTM模型
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim  # 隐藏层维度
        self.num_layers = num_layers  # LSTM层的数量
        self.output_dim = output_dim

        # LSTM网络层
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        # 全连接层，用于将LSTM的输出转换为最终的输出维度
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, src, tgt=None):
        # src: [batch_size, seq_len, input_dim]
        # tgt: 仅为了与Informer接口兼容，实际不使用
        
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, src.size(0), self.hidden_dim).requires_grad_().to(src.device)
        c0 = torch.zeros(self.num_layers, src.size(0), self.hidden_dim).requires_grad_().to(src.device)
        # 前向传播LSTM，返回输出和最新的隐藏状态与细胞状态
        out, (hn, cn) = self.lstm(src, (h0.detach(), c0.detach()))
        # 将LSTM的最后一个时间步的输出通过全连接层
        out = self.fc(out[:, -1, :])
        # 增加一个维度以匹配Informer的输出格式
        out = out.unsqueeze(1)
        return out


# GRU模型
class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(GRU, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim

        # GRU网络层
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        # 全连接层，与LSTM相同
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, src, tgt=None):
        # src: [batch_size, seq_len, input_dim]
        # tgt: 仅为了与Informer接口兼容，实际不使用
        
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, src.size(0), self.hidden_dim).requires_grad_().to(src.device)
        # 前向传播GRU
        out, hn = self.gru(src, h0.detach())
        # 将GRU的最后一个时间步的输出通过全连接层
        out = self.fc(out[:, -1, :])
        # 增加一个维度以匹配Informer的输出格式
        out = out.unsqueeze(1)
        return out


# 双向LSTM模型
class BiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(BiLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim

        # 双向LSTM网络层
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        # 注意：因为是双向，所以全连接层的输入是 hidden_dim * 2
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, src, tgt=None):
        # src: [batch_size, seq_len, input_dim]
        # tgt: 仅为了与Informer接口兼容，实际不使用
        
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers * 2, src.size(0), self.hidden_dim).requires_grad_().to(src.device)
        c0 = torch.zeros(self.num_layers * 2, src.size(0), self.hidden_dim).requires_grad_().to(src.device)
        # 前向传播双向LSTM
        out, (hn, cn) = self.lstm(src, (h0.detach(), c0.detach()))
        # 取双向的最后一个时间步的输出
        out = self.fc(out[:, -1, :])
        # 增加一个维度以匹配Informer的输出格式
        out = out.unsqueeze(1)
        return out


# TCN模型
class TemporalConvNet(nn.Module):
    def __init__(self, input_dim, output_dim, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_dim if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size, 
                                   padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], output_dim)

    def forward(self, src, tgt=None):
        # src: [batch_size, seq_len, input_dim]
        # tgt: 仅为了与Informer接口兼容，实际不使用
        
        src = src.transpose(1, 2)  # 将 batch_size, seq_len, input_dim 转换为 batch_size, input_dim, seq_len
        out = self.network(src)
        out = out[:, :, -1]  # 选择每个序列的最后一个输出
        out = self.fc(out)
        # 增加一个维度以匹配Informer的输出格式
        out = out.unsqueeze(1)
        return out


class TemporalBlock(nn.Module):
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


class Chomp1d(nn.Module):
    def __init__(self, padding):
        super(Chomp1d, self).__init__()
        self.padding = padding

    def forward(self, x):
        # Chomp1d 是一个简单的自定义层，用于剪切掉因为填充(padding)导致的多余的输出，
        # 这是保证因果卷积不看到未来信息的关键。
        return x[:, :, :-self.padding]


# Transformer模型
class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, output_dim, hidden_dim, dropout_rate=0.1):
        super(TimeSeriesTransformer, self).__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        # Transformer 的 Encoder 部分
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,  # 输入特征维度
            nhead=num_heads,  # 多头注意力机制的头数
            dropout=dropout_rate
        )
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)

        # 将 Encoder 的输出通过一个全连接层转换为所需的输出维度
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.transform_layer = nn.Linear(input_dim, hidden_dim)
        
    def forward(self, src, tgt=None):
        # src: [batch_size, seq_len, input_dim]
        # tgt: 仅为了与Informer接口兼容，实际不使用
        
        # 转换输入数据维度以符合 Transformer 的要求：(seq_len, batch_size, feature_dim)
        src = src.permute(1, 0, 2)
        src = self.transform_layer(src)
        # Transformer 编码器
        out = self.transformer_encoder(src)

        # 取最后一个时间步的输出
        out = out[-1, :, :]

        # 全连接层生成最终输出
        out = self.output_layer(out)
        # 增加一个维度以匹配Informer的输出格式
        out = out.unsqueeze(1)
        return out


# 没有decoder的Informer
class Informer(nn.Module):
    def __init__(self, input_dim, output_dim, d_model=512, n_heads=8, d_ff=2048,
                 enc_layers=3, dec_layers=2, dropout=0.1):
        super(Informer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.d_model = d_model

        # 使用PyTorch内置的Transformer模块作为替代
        # 编码器使用TransformerEncoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, enc_layers)

        # 输入投影层
        self.input_projection = nn.Linear(input_dim, d_model)

        # 由于我们只需要编码器输出，移除解码器
        # 输出投影层
        self.intermediate = nn.Linear(d_model, d_model // 2)
        self.activation = nn.ReLU()
        self.projection = nn.Linear(d_model // 2, output_dim)

    def forward(self, src, tgt=None):
        """
        src: [batch_size, src_seq_len, input_dim] - 输入序列（例如历史数据）
        tgt: 仅为了与接口兼容，实际不使用

        这个简化版本的Informer只使用编码器部分，类似于Transformer模型的实现方式。
        """
        # src: [batch_size, src_seq_len, input_dim]
        # tgt: 仅为了与接口兼容，实际不使用

        # 输入投影
        src_emb = self.input_projection(src)

        # 位置编码可以使用PyTorch内置的
        src_emb = src_emb * math.sqrt(self.d_model)

        # 编码器前向传播
        enc_output = self.encoder(src_emb)

        # 取最后一个时间步的输出（类似于Transformer模型的做法）
        enc_output = enc_output[:, -1, :]

        # 最终输出投影
        intermediate_output = self.intermediate(enc_output)
        intermediate_output = self.activation(intermediate_output)
        output = self.projection(intermediate_output)

        # 增加一个维度以匹配输出格式
        output = output.unsqueeze(1)

        return output