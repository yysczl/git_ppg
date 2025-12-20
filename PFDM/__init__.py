"""
PPG-Former-DualStream 项目
融合多尺度时频Transformer与双流协同网络的多任务心理压力预测

项目结构:
- config.py: 配置文件，包含模型参数和训练参数
- models.py: 模型文件，包含PPG-Former、Dual-Stream等独立模块
- trainer.py: 训练文件，包含K折交叉验证训练逻辑
- evaluator.py: 评估文件，包含模型评估和测试代码
- utils.py: 工具文件，包含数据加载、绘图、日志等功能
- main.py: 主程序入口，整合所有模块

目录结构:
- logs/: 训练日志
- checkpoints/: 模型检查点
- results/: 结果和图表

使用方法:
    python main.py                          # 使用默认配置训练
    python main.py --config ppg_only        # 仅使用PPG模型训练
    python main.py --config ablation_no_pe  # 消融实验
    python main.py --mode eval --model_path checkpoints/best_model.pth  # 评估模式
"""

from .config import (
    ExperimentConfig,
    ModelConfig,
    TrainingConfig,
    DataConfig,
    LogConfig,
    AblationConfig,
    get_config,
    get_default_config
)

from .models import (
    # 位置编码
    StandardPositionalEncoding,
    PhysiologicalPositionalEncoding,
    # PPG-Former模块
    MultiScaleConvBlock,
    TimeFreqAttention,
    StressAwareGating,
    PPGFormerBlock,
    PPGFormerEncoder,
    # PRV编码器
    PRVEncoder,
    # 双流融合模块
    CrossModalAttention,
    DualStreamFusion,
    # 多任务头
    MultiTaskHead,
    # 完整模型
    PPGFormer,
    PPGFormerDualStream,
    DualStreamOnly,
    # 基线模型
    LSTMBaseline,
    TransformerBaseline,
    # 工厂函数
    create_model
)

from .trainer import (
    Trainer,
    train_single_fold,
    train_kfold,
    get_average_history,
    get_best_fold
)

from .evaluator import (
    Evaluator,
    evaluate_model,
    compare_models,
    ablation_study,
    ResultAnalyzer
)

from .utils import (
    set_seed,
    Logger,
    PPGDataset,
    MultiModalDataset,
    load_ppg_data,
    load_prv_data,
    create_data_loaders,
    plot_training_process,
    plot_predictions,
    plot_fold_comparison,
    count_parameters,
    save_model,
    load_model,
    calculate_metrics
)

__version__ = '1.0.0'
__author__ = 'PPG-Former Team'
