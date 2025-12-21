"""
PPG-Former-DualStream 项目
融合多尺度时频Transformer与双流协同网络的多任务心理压力预测

项目结构:
- config.py: 配置文件，包含模型参数、训练模式和任务类型
- models.py: 模型文件，包含PPGFormer、PRVModel、PPGFormerDualStream等独立模块
- trainer.py: 训练文件，包含单任务/多任务/K折交叉验证训练逻辑
- evaluator.py: 评估文件，包含MAE、RMSE、Accuracy、Precision、Recall、F1-Score等评估指标
- utils.py: 工具文件，包含数据加载、情绪标签处理、日志等功能
- main.py: 主程序入口，整合所有模块

训练模式:
- ppg_only: 仅PPG训练
- prv_only: 仅PRV训练
- dual_stream: PPG+PRV双流训练
- ppg_regression: PPG单独压力回归
- prv_regression: PRV单独压力回归
- ppg_classification: PPG单独情绪分类
- prv_classification: PRV单独情绪分类
- multi_task: 多任务训练（压力回归+情绪分类）

基准模型训练模式:
- baseline_ppg_regression: 基准模型PPG压力回归
- baseline_prv_regression: 基准模型PRV压力回归
- baseline_ppg_classification: 基准模型PPG情绪分类
- baseline_prv_classification: 基准模型PRV情绪分类

可用基准模型:
- lstm: LSTM模型
- gru: GRU模型
- bilstm: 双向LSTM模型
- tcn: 时序卷积网络
- transformer_baseline: Transformer基线模型
- informer: Informer模型

使用方法:
    # PPG独立压力回归
    python main.py --mode train --train_mode ppg_regression
    
    # PRV情绪分类
    python main.py --mode train --train_mode prv_classification
    
    # 双流多任务训练
    python main.py --mode train --train_mode multi_task
    
    # 基准模型训练
    python main.py --mode train --train_mode baseline_ppg_regression --baseline_model lstm
    python main.py --mode train --train_mode baseline_prv_classification --baseline_model informer
    
    # 消融实验
    python main.py --mode ablation
"""

from .config import (
    ExperimentConfig,
    ModelConfig,
    BaselineModelConfig,
    TrainingConfig,
    DataConfig,
    LogConfig,
    AblationConfig,
    TrainMode,
    TaskType,
    get_config,
    get_default_config,
    list_available_configs,
    list_baseline_models,
    AVAILABLE_BASELINE_MODELS
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
    # 任务头
    RegressionHead,
    ClassificationHead,
    MultiTaskHead,
    # 完整模型
    PPGFormer,
    PRVModel,
    PPGFormerDualStream,
    DualStreamOnly,
    # 基线模型
    LSTMBaseline,
    TransformerBaseline,
    # 基准模型（支持回归/分类任务）
    LSTMModel,
    GRUModel,
    BiLSTMModel,
    TCNModel,
    TransformerModel,
    InformerModel,
    BASELINE_MODELS,
    # 工厂函数
    create_model,
    create_baseline_model,
    get_model_for_train_mode,
    list_available_models
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
    compare_models
)

from .utils import (
    set_seed,
    Logger,
    # 情绪标签
    EMOTION_NAMES,
    EMOTION_LABEL_MAP,
    # 数据集类
    PPGDataset,
    MultiModalDataset,
    # 数据加载函数
    load_emotion_data_from_folder,
    load_all_emotion_data,
    prepare_data_for_training,
    split_data_by_emotion,
    load_single_emotion_data,
    load_ppg_data,
    load_prv_data,
    create_data_loaders,
    # 绘图函数
    plot_training_process,
    plot_predictions,
    plot_fold_comparison,
    # 模型工具
    count_parameters,
    save_model,
    load_model,
    # 评估指标
    calculate_metrics,
    calculate_classification_metrics
)

__version__ = '2.1.0'
__author__ = 'PPG-Former Team'
