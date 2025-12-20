"""
PPG-Former-DualStream 配置文件
包含模型参数、训练参数和实验配置
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:
    """模型配置"""
    # PPG-Former参数
    ppg_input_dim: int = 1
    prv_input_dim: int = 1
    d_model: int = 128
    n_heads: int = 8
    d_ff: int = 512
    ppg_layers: int = 3
    prv_layers: int = 2
    fusion_layers: int = 2
    num_emotions: int = 5
    scales: List[int] = field(default_factory=lambda: [1, 3, 5, 7])
    dropout: float = 0.1
    
    # 生理周期位置编码参数
    max_len: int = 5000
    heart_rate_range: tuple = (60, 100)  # bpm范围


@dataclass
class TrainingConfig:
    """训练配置"""
    # 基础训练参数
    num_epochs: int = 300
    batch_size: int = 8
    learning_rate: float = 0.0005
    weight_decay: float = 1e-4
    
    # K折交叉验证
    k_folds: int = 5
    
    # 学习率调度器
    scheduler_factor: float = 0.5
    scheduler_patience: int = 20
    
    # 梯度裁剪
    max_grad_norm: float = 1.0
    
    # 是否使用情绪分类任务
    use_emotion: bool = False
    
    # 随机种子
    seed: int = 42


@dataclass
class DataConfig:
    """数据配置"""
    # 数据集路径
    ppg_file_path: str = "../dataset/PPG/StressPPG.csv"
    prv_file_path: str = "../dataset/PRV/StressPRV.csv"
    
    # PPG数据参数
    ppg_seq_len: int = 1800
    
    # PRV数据参数
    prv_seq_len: int = 80
    
    # 数据划分比例
    test_size: float = 0.1
    val_size: float = 0.1


@dataclass
class LogConfig:
    """日志配置"""
    # 日志目录
    log_dir: str = "logs"
    
    # 模型保存目录
    model_save_dir: str = "checkpoints"
    
    # 结果保存目录
    result_dir: str = "results"
    
    # 日志级别
    log_level: str = "INFO"
    
    # 是否保存训练过程图
    save_plots: bool = True
    
    # 打印频率（每多少个epoch打印一次）
    print_freq: int = 10


@dataclass
class AblationConfig:
    """消融实验配置"""
    # 是否使用生理周期位置编码
    use_physiological_pe: bool = True
    
    # 是否使用多尺度卷积
    use_multi_scale_conv: bool = True
    
    # 是否使用时频融合注意力
    use_time_freq_attention: bool = True
    
    # 是否使用频域注意力分支
    use_freq_attention: bool = True
    
    # 是否使用压力感知门控
    use_stress_gating: bool = True
    
    # 是否使用跨模态交互注意力
    use_cross_modal_attention: bool = True
    
    # 是否使用不确定性加权损失
    use_uncertainty_weighting: bool = True
    
    # 是否使用双流融合
    use_dual_stream: bool = True


@dataclass
class ExperimentConfig:
    """实验总配置"""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    log: LogConfig = field(default_factory=LogConfig)
    ablation: AblationConfig = field(default_factory=AblationConfig)
    
    # 实验名称
    experiment_name: str = "PPGFormerDualStream"
    
    # 设备
    device: str = "cuda"  # 或 "cpu"
    
    def __post_init__(self):
        """初始化后处理"""
        # 创建必要的目录
        os.makedirs(self.log.log_dir, exist_ok=True)
        os.makedirs(self.log.model_save_dir, exist_ok=True)
        os.makedirs(self.log.result_dir, exist_ok=True)


# 预定义配置
def get_default_config() -> ExperimentConfig:
    """获取默认配置"""
    return ExperimentConfig()


def get_ppg_only_config() -> ExperimentConfig:
    """获取仅PPG模型配置"""
    config = ExperimentConfig()
    config.experiment_name = "PPGFormer"
    config.ablation.use_dual_stream = False
    return config


def get_ablation_no_pe_config() -> ExperimentConfig:
    """消融实验：不使用生理周期位置编码"""
    config = ExperimentConfig()
    config.experiment_name = "Ablation_NoPE"
    config.ablation.use_physiological_pe = False
    return config


def get_ablation_no_freq_config() -> ExperimentConfig:
    """消融实验：不使用频域注意力"""
    config = ExperimentConfig()
    config.experiment_name = "Ablation_NoFreq"
    config.ablation.use_freq_attention = False
    return config


def get_ablation_no_cross_attn_config() -> ExperimentConfig:
    """消融实验：不使用跨模态交互注意力"""
    config = ExperimentConfig()
    config.experiment_name = "Ablation_NoCrossAttn"
    config.ablation.use_cross_modal_attention = False
    return config


def get_ablation_no_uncertainty_config() -> ExperimentConfig:
    """消融实验：不使用不确定性加权"""
    config = ExperimentConfig()
    config.experiment_name = "Ablation_NoUncertainty"
    config.ablation.use_uncertainty_weighting = False
    return config


# 所有预定义配置
CONFIGS = {
    "default": get_default_config,
    "ppg_only": get_ppg_only_config,
    "ablation_no_pe": get_ablation_no_pe_config,
    "ablation_no_freq": get_ablation_no_freq_config,
    "ablation_no_cross_attn": get_ablation_no_cross_attn_config,
    "ablation_no_uncertainty": get_ablation_no_uncertainty_config,
}


def get_config(name: str = "default") -> ExperimentConfig:
    """根据名称获取配置"""
    if name not in CONFIGS:
        raise ValueError(f"未知配置名称: {name}. 可用配置: {list(CONFIGS.keys())}")
    return CONFIGS[name]()
