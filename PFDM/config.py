"""PPG-Former-DualStream 配置文件
包含模型参数、训练参数和实验配置
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum


class TrainMode(Enum):
    """训练模式枚举"""
    # 单模态训练
    PPG_ONLY = "ppg_only"              # 仅PPG训练
    PRV_ONLY = "prv_only"              # 仅PRV训练
    
    # 双流训练
    DUAL_STREAM = "dual_stream"        # PPG + PRV双流训练
    
    # 单任务训练
    PPG_REGRESSION = "ppg_regression"  # PPG压力回归
    PRV_REGRESSION = "prv_regression"  # PRV压力回归
    PPG_CLASSIFICATION = "ppg_classification"  # PPG情绪分类
    PRV_CLASSIFICATION = "prv_classification"  # PRV情绪分类
    
    # 多任务训练
    MULTI_TASK = "multi_task"          # 压力回归 + 情绪分类
    
    # 基准模型训练（四种单任务模式）
    BASELINE_PPG_REGRESSION = "baseline_ppg_regression"      # 基准模型PPG压力回归
    BASELINE_PRV_REGRESSION = "baseline_prv_regression"      # 基准模型PRV压力回归
    BASELINE_PPG_CLASSIFICATION = "baseline_ppg_classification"  # 基准模型PPG情绪分类
    BASELINE_PRV_CLASSIFICATION = "baseline_prv_classification"  # 基准模型PRV情绪分类


class TaskType(Enum):
    """任务类型枚举"""
    REGRESSION = "regression"          # 回归任务
    CLASSIFICATION = "classification"  # 分类任务
    MULTI_TASK = "multi_task"          # 多任务


@dataclass
class BaselineModelConfig:
    """基准模型配置
    
    与informer-prv保持一致的参数设置
    """
    # 基准模型名称: lstm, gru, bilstm, tcn, transformer_baseline, informer
    model_name: str = "lstm"
    
    # 通用参数（与informer-prv一致）
    input_dim: int = 1
    hidden_dim: int = 64  # 128
    num_layers: int = 2
    num_classes: int = 5
    dropout: float = 0.3   # 0.1
    
    # TCN专用参数
    kernel_size: int = 3
    
    # Transformer/Informer专用参数
    n_heads: int = 8


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
    """训练配置
    
    与informer-prv保持一致的训练参数
    """
    # 基础训练参数（与informer-prv一致）
    num_epochs: int = 300       # 从informer-prv: 300
    batch_size: int = 16         # 从informer-prv: 8
    learning_rate: float = 0.001  # 从informer-prv: 0.0005
    weight_decay: float = 1e-3  # 从informer-prv: 1e-4
    
    # K折交叉验证
    k_folds: int = 5            # 从informer-prv: 5
    
    # 学习率调度器（与informer-prv一致）
    scheduler_factor: float = 0.5    # 从informer-prv: 0.5
    scheduler_patience: int = 30     # 从informer-prv: 20
    
    # 梯度裁剪
    use_grad_clip: bool = False  # 是否启用梯度裁剪
    max_grad_norm: float = 1.0  # 梯度裁剪最大范数
    
    # 训练模式
    train_mode: str = "dual_stream"    # 训练模式
    task_type: str = "regression"      # 任务类型: regression/classification/multi_task
    
    # 是否使用情绪分类任务
    use_emotion: bool = False
    
    # 随机种子
    seed: int = 42
    
    # 早停配置
    early_stopping: bool = False
    early_stopping_patience: int = 50


@dataclass
class DataConfig:
    """数据配置"""
    # 数据集根目录
    data_root: str = "dataset"
    
    # PPG数据目录
    ppg_dir: str = "PPG"
    
    # PRV数据目录
    prv_dir: str = "PRV"
    
    # 情绪类别列表
    emotion_categories: List[str] = field(default_factory=lambda: [
        "Anxiety", "Happy", "Peace", "Sad", "Stress"
    ])
    
    # 情绪标签映射 (文件名 -> 数字标签)
    emotion_label_map: dict = field(default_factory=lambda: {
        "Anxiety": 0, "Happy": 1, "Peace": 2, "Sad": 3, "Stress": 4
    })
    
    # 选择使用的情绪类别（None表示使用全部）
    selected_emotions: Optional[List[str]] = None
    
    # 压力回归任务的目标情绪类别（用于单任务压力回归训练）
    # 当进行压力回归单任务时，可以指定只使用某一种情绪类别的数据
    # None表示使用selected_emotions或全部情绪数据
    target_emotion_for_regression: Optional[str] = None
    
    # PPG数据参数
    ppg_seq_len: int = 1800
    
    # PRV数据参数
    prv_seq_len: int = 80
    
    # 每个情绪文件的样本数
    samples_per_emotion: int = 90
    
    # 数据划分比例
    test_size: float = 0.1
    val_size: float = 0.1
    
    @property
    def ppg_data_path(self) -> str:
        """获取PPG数据目录完整路径"""
        return os.path.join(self.data_root, self.ppg_dir)
    
    @property
    def prv_data_path(self) -> str:
        """获取PRV数据目录完整路径"""
        return os.path.join(self.data_root, self.prv_dir)
    
    def get_emotions_for_task(self, task_type: str) -> List[str]:
        """
        根据任务类型获取应使用的情绪类别
        
        Args:
            task_type: 任务类型 ('regression', 'classification', 'multi_task')
        
        Returns:
            情绪类别列表
        
        规则:
        - 情绪分类任务: 必须使用全部5种情绪类别
        - 压力回归任务: 可以使用指定的目标情绪类别或全部情绪类别
        - 多任务: 必须使用全部5种情绪类别
        """
        if task_type in ['classification', 'multi_task']:
            # 情绪分类和多任务必须使用全部情绪类别
            return self.emotion_categories.copy()
        elif task_type == 'regression':
            # 压力回归任务可以使用特定情绪类别
            if self.target_emotion_for_regression is not None:
                # 验证目标情绪是否有效
                if self.target_emotion_for_regression not in self.emotion_categories:
                    raise ValueError(
                        f"无效的目标情绪类别: {self.target_emotion_for_regression}. "
                        f"可用类别: {self.emotion_categories}"
                    )
                return [self.target_emotion_for_regression]
            elif self.selected_emotions is not None:
                return self.selected_emotions
            else:
                return self.emotion_categories.copy()
        else:
            # 默认返回全部情绪类别
            return self.emotion_categories.copy()


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
    baseline_model: BaselineModelConfig = field(default_factory=BaselineModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    log: LogConfig = field(default_factory=LogConfig)
    ablation: AblationConfig = field(default_factory=AblationConfig)
    
    # 实验名称
    experiment_name: str = "PPGFormerDualStream"
    
    # 设备
    device: str = "cuda"  # 或 "cpu"
    
    # 是否使用基准模型
    use_baseline_model: bool = False
    
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


# ============ 单任务训练配置 ============
def get_ppg_regression_config() -> ExperimentConfig:
    """PPG单独压力回归配置"""
    config = ExperimentConfig()
    config.experiment_name = "PPG_Regression"
    config.training.train_mode = "ppg_regression"
    config.training.task_type = "regression"
    config.ablation.use_dual_stream = False
    return config


def get_prv_regression_config() -> ExperimentConfig:
    """PRV单独压力回归配置"""
    config = ExperimentConfig()
    config.experiment_name = "PRV_Regression"
    config.training.train_mode = "prv_regression"
    config.training.task_type = "regression"
    config.ablation.use_dual_stream = False
    return config


def get_ppg_classification_config() -> ExperimentConfig:
    """PPG单独情绪分类配置"""
    config = ExperimentConfig()
    config.experiment_name = "PPG_Classification"
    config.training.train_mode = "ppg_classification"
    config.training.task_type = "classification"
    config.training.use_emotion = True
    config.ablation.use_dual_stream = False
    return config


def get_prv_classification_config() -> ExperimentConfig:
    """PRV单独情绪分类配置"""
    config = ExperimentConfig()
    config.experiment_name = "PRV_Classification"
    config.training.train_mode = "prv_classification"
    config.training.task_type = "classification"
    config.training.use_emotion = True
    config.ablation.use_dual_stream = False
    return config


def get_multi_task_config() -> ExperimentConfig:
    """多任务训练配置（压力回归+情绪分类）"""
    config = ExperimentConfig()
    config.experiment_name = "MultiTask"
    config.training.train_mode = "multi_task"
    config.training.task_type = "multi_task"
    config.training.use_emotion = True
    config.ablation.use_dual_stream = True
    return config


def get_prv_only_config() -> ExperimentConfig:
    """仅PRV模型配置"""
    config = ExperimentConfig()
    config.experiment_name = "PRVOnly"
    config.training.train_mode = "prv_only"
    config.ablation.use_dual_stream = False
    return config


# ============ 基准模型训练配置 ============
def get_baseline_ppg_regression_config(model_name: str = "lstm") -> ExperimentConfig:
    """基准模型PPG压力回归配置"""
    config = ExperimentConfig()
    config.experiment_name = f"Baseline_{model_name.upper()}_PPG_Regression"
    config.training.train_mode = "baseline_ppg_regression"
    config.training.task_type = "regression"
    config.use_baseline_model = True
    config.baseline_model.model_name = model_name
    config.ablation.use_dual_stream = False
    return config


def get_baseline_prv_regression_config(model_name: str = "lstm") -> ExperimentConfig:
    """基准模型PRV压力回归配置"""
    config = ExperimentConfig()
    config.experiment_name = f"Baseline_{model_name.upper()}_PRV_Regression"
    config.training.train_mode = "baseline_prv_regression"
    config.training.task_type = "regression"
    config.use_baseline_model = True
    config.baseline_model.model_name = model_name
    config.ablation.use_dual_stream = False
    return config


def get_baseline_ppg_classification_config(model_name: str = "lstm") -> ExperimentConfig:
    """基准模型PPG情绪分类配置"""
    config = ExperimentConfig()
    config.experiment_name = f"Baseline_{model_name.upper()}_PPG_Classification"
    config.training.train_mode = "baseline_ppg_classification"
    config.training.task_type = "classification"
    config.training.use_emotion = True
    config.use_baseline_model = True
    config.baseline_model.model_name = model_name
    config.ablation.use_dual_stream = False
    return config


def get_baseline_prv_classification_config(model_name: str = "lstm") -> ExperimentConfig:
    """基准模型PRV情绪分类配置"""
    config = ExperimentConfig()
    config.experiment_name = f"Baseline_{model_name.upper()}_PRV_Classification"
    config.training.train_mode = "baseline_prv_classification"
    config.training.task_type = "classification"
    config.training.use_emotion = True
    config.use_baseline_model = True
    config.baseline_model.model_name = model_name
    config.ablation.use_dual_stream = False
    return config


# 所有可用的基准模型名称
AVAILABLE_BASELINE_MODELS = ["lstm", "gru", "bilstm", "tcn", "transformer_baseline", "informer"]


# 所有预定义配置
CONFIGS = {
    "default": get_default_config,
    "ppg_only": get_ppg_only_config,
    "prv_only": get_prv_only_config,
    "ppg_regression": get_ppg_regression_config,
    "prv_regression": get_prv_regression_config,
    "ppg_classification": get_ppg_classification_config,
    "prv_classification": get_prv_classification_config,
    "multi_task": get_multi_task_config,
    "ablation_no_pe": get_ablation_no_pe_config,
    "ablation_no_freq": get_ablation_no_freq_config,
    "ablation_no_cross_attn": get_ablation_no_cross_attn_config,
    "ablation_no_uncertainty": get_ablation_no_uncertainty_config,
    # 基准模型配置（使用默认LSTM）
    "baseline_ppg_regression": get_baseline_ppg_regression_config,
    "baseline_prv_regression": get_baseline_prv_regression_config,
    "baseline_ppg_classification": get_baseline_ppg_classification_config,
    "baseline_prv_classification": get_baseline_prv_classification_config,
}


def get_config(name: str = "default", baseline_model: str = None) -> ExperimentConfig:
    """根据名称获取配置
    
    Args:
        name: 配置名称
        baseline_model: 基准模型名称（仅用于基准模型配置）
    
    Returns:
        ExperimentConfig实例
    """
    if name not in CONFIGS:
        raise ValueError(f"未知配置名称: {name}. 可用配置: {list(CONFIGS.keys())}")
    
    config_func = CONFIGS[name]
    
    # 检查是否是基准模型配置
    if name.startswith("baseline_") and baseline_model:
        if baseline_model not in AVAILABLE_BASELINE_MODELS:
            raise ValueError(f"未知基准模型: {baseline_model}. 可用模型: {AVAILABLE_BASELINE_MODELS}")
        return config_func(baseline_model)
    
    return config_func()


def list_available_configs() -> List[str]:
    """列出所有可用配置"""
    return list(CONFIGS.keys())


def list_baseline_models() -> List[str]:
    """列出所有可用的基准模型"""
    return AVAILABLE_BASELINE_MODELS
