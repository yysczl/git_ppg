"""PRV Informer 配置文件
包含模型参数、训练参数和实验配置
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum


class ModelType(Enum):
    """模型类型枚举"""
    LSTM = "lstm"
    GRU = "gru"
    BILSTM = "bilstm"
    TCN = "tcn"
    TRANSFORMER = "transformer"
    INFORMER = "informer"


class SignalType(Enum):
    """信号类型枚举"""
    PRV = "prv"
    PPG = "ppg"
    PUPIL = "pupil"


@dataclass
class LSTMConfig:
    """LSTM模型配置"""
    input_dim: int = 1
    hidden_dim: int = 128
    num_layers: int = 2
    output_dim: int = 1


@dataclass
class GRUConfig:
    """GRU模型配置"""
    input_dim: int = 1
    hidden_dim: int = 128
    num_layers: int = 2
    output_dim: int = 1


@dataclass
class BiLSTMConfig:
    """BiLSTM模型配置"""
    input_dim: int = 1
    hidden_dim: int = 128
    num_layers: int = 2
    output_dim: int = 1


@dataclass
class TCNConfig:
    """TCN模型配置"""
    input_dim: int = 1
    output_dim: int = 1
    num_channels: List[int] = field(default_factory=lambda: [64, 64, 128, 128])
    kernel_size: int = 3
    dropout: float = 0.1


@dataclass
class TransformerConfig:
    """Transformer模型配置"""
    input_dim: int = 1
    num_heads: int = 8
    num_layers: int = 3
    output_dim: int = 1
    hidden_dim: int = 128
    dropout_rate: float = 0.1


@dataclass
class InformerConfig:
    """Informer模型配置"""
    input_dim: int = 1
    output_dim: int = 1
    d_model: int = 128
    n_heads: int = 8
    d_ff: int = 512
    enc_layers: int = 3
    dec_layers: int = 2
    dropout: float = 0.1


@dataclass
class TrainingConfig:
    """训练配置"""
    # 基础训练参数
    num_epochs: int = 100
    batch_size: int = 8
    learning_rate: float = 0.0005
    weight_decay: float = 1e-4
    
    # K折交叉验证
    k_folds: int = 5
    
    # 学习率调度器
    scheduler_factor: float = 0.5
    scheduler_patience: int = 20
    
    # 随机种子
    seed: int = 42


@dataclass
class DataConfig:
    """数据配置"""
    # 数据集根目录
    data_root: str = "../../PFDM/dataset/PRV"
    
    # 数据文件名
    data_file: str = "Stress.csv"
    
    # 信号类型
    signal_type: str = "prv"
    
    # 数据划分比例
    test_size: float = 0.1
    val_size: float = 0.1
    
    @property
    def data_path(self) -> str:
        """获取数据文件完整路径"""
        return os.path.join(self.data_root, self.data_file)


@dataclass
class LogConfig:
    """日志配置"""
    # 模型保存目录
    model_save_dir: str = "checkpoints"
    
    # 结果保存目录
    result_dir: str = "results"
    
    # 是否保存训练过程图
    save_plots: bool = True
    
    # 打印频率（每多少个epoch打印一次）
    print_freq: int = 10


@dataclass
class ExperimentConfig:
    """实验总配置"""
    # 模型配置
    lstm: LSTMConfig = field(default_factory=LSTMConfig)
    gru: GRUConfig = field(default_factory=GRUConfig)
    bilstm: BiLSTMConfig = field(default_factory=BiLSTMConfig)
    tcn: TCNConfig = field(default_factory=TCNConfig)
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    informer: InformerConfig = field(default_factory=InformerConfig)
    
    # 训练配置
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # 数据配置
    data: DataConfig = field(default_factory=DataConfig)
    
    # 日志配置
    log: LogConfig = field(default_factory=LogConfig)
    
    # 当前使用的模型类型
    model_type: str = "lstm"
    
    # 实验名称
    experiment_name: str = "PRV_Stress_Prediction"
    
    # 设备
    device: str = "cuda"
    
    def __post_init__(self):
        """初始化后处理"""
        os.makedirs(self.log.model_save_dir, exist_ok=True)
        os.makedirs(self.log.result_dir, exist_ok=True)
    
    def get_model_config(self) -> dict:
        """根据模型类型获取模型参数"""
        model_type = self.model_type.lower()
        
        if model_type == "lstm":
            cfg = self.lstm
            return {
                'input_dim': cfg.input_dim,
                'hidden_dim': cfg.hidden_dim,
                'num_layers': cfg.num_layers,
                'output_dim': cfg.output_dim
            }
        elif model_type == "gru":
            cfg = self.gru
            return {
                'input_dim': cfg.input_dim,
                'hidden_dim': cfg.hidden_dim,
                'num_layers': cfg.num_layers,
                'output_dim': cfg.output_dim
            }
        elif model_type == "bilstm":
            cfg = self.bilstm
            return {
                'input_dim': cfg.input_dim,
                'hidden_dim': cfg.hidden_dim,
                'num_layers': cfg.num_layers,
                'output_dim': cfg.output_dim
            }
        elif model_type == "tcn":
            cfg = self.tcn
            return {
                'input_dim': cfg.input_dim,
                'output_dim': cfg.output_dim,
                'num_channels': cfg.num_channels,
                'kernel_size': cfg.kernel_size,
                'dropout': cfg.dropout
            }
        elif model_type == "transformer":
            cfg = self.transformer
            return {
                'input_dim': cfg.input_dim,
                'num_heads': cfg.num_heads,
                'num_layers': cfg.num_layers,
                'output_dim': cfg.output_dim,
                'hidden_dim': cfg.hidden_dim,
                'dropout_rate': cfg.dropout_rate
            }
        elif model_type == "informer":
            cfg = self.informer
            return {
                'input_dim': cfg.input_dim,
                'output_dim': cfg.output_dim,
                'd_model': cfg.d_model,
                'n_heads': cfg.n_heads,
                'd_ff': cfg.d_ff,
                'enc_layers': cfg.enc_layers,
                'dec_layers': cfg.dec_layers,
                'dropout': cfg.dropout
            }
        else:
            raise ValueError(f"未知模型类型: {model_type}")


# ============ 预定义配置函数 ============

def get_default_config() -> ExperimentConfig:
    """获取默认配置"""
    return ExperimentConfig()


def get_lstm_config() -> ExperimentConfig:
    """获取LSTM配置"""
    config = ExperimentConfig()
    config.model_type = "lstm"
    config.experiment_name = "LSTM_PRV_Prediction"
    return config


def get_gru_config() -> ExperimentConfig:
    """获取GRU配置"""
    config = ExperimentConfig()
    config.model_type = "gru"
    config.experiment_name = "GRU_PRV_Prediction"
    return config


def get_bilstm_config() -> ExperimentConfig:
    """获取BiLSTM配置"""
    config = ExperimentConfig()
    config.model_type = "bilstm"
    config.experiment_name = "BiLSTM_PRV_Prediction"
    return config


def get_tcn_config() -> ExperimentConfig:
    """获取TCN配置"""
    config = ExperimentConfig()
    config.model_type = "tcn"
    config.experiment_name = "TCN_PRV_Prediction"
    return config


def get_transformer_config() -> ExperimentConfig:
    """获取Transformer配置"""
    config = ExperimentConfig()
    config.model_type = "transformer"
    config.experiment_name = "Transformer_PRV_Prediction"
    return config


def get_informer_config() -> ExperimentConfig:
    """获取Informer配置"""
    config = ExperimentConfig()
    config.model_type = "informer"
    config.experiment_name = "Informer_PRV_Prediction"
    return config


# 所有预定义配置
CONFIGS = {
    "default": get_default_config,
    "lstm": get_lstm_config,
    "gru": get_gru_config,
    "bilstm": get_bilstm_config,
    "tcn": get_tcn_config,
    "transformer": get_transformer_config,
    "informer": get_informer_config,
}

# 所有可用的模型类型
AVAILABLE_MODELS = ["lstm", "gru", "bilstm", "tcn", "transformer", "informer"]


def get_config(name: str = "default") -> ExperimentConfig:
    """根据名称获取配置
    
    Args:
        name: 配置名称
    
    Returns:
        ExperimentConfig实例
    """
    if name not in CONFIGS:
        raise ValueError(f"未知配置名称: {name}. 可用配置: {list(CONFIGS.keys())}")
    
    return CONFIGS[name]()


def list_available_configs() -> List[str]:
    """列出所有可用配置"""
    return list(CONFIGS.keys())


def list_available_models() -> List[str]:
    """列出所有可用的模型"""
    return AVAILABLE_MODELS
