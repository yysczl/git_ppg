"""PPG-Former-DualStream 主程序入口
融合多尺度时频Transformer与双流协同的多任务心理压力预测

================================================================================
使用方法
================================================================================

1. 双流模型训练
--------------------------------------------------------------------------------
    # 双流融合训练（PPG + PRV）
    python main.py --mode train --train_mode dual_stream
    
    # PPG单流训练
    python main.py --mode train --train_mode ppg_only
    
    # PRV单流训练
    python main.py --mode train --train_mode prv_only

2. 压力回归单任务训练
--------------------------------------------------------------------------------
    # PPG压力回归（使用全部情绪数据）
    python main.py --mode train --train_mode ppg_regression
    
    # PRV压力回归（使用全部情绪数据）
    python main.py --mode train --train_mode prv_regression
    
    # 指定单一目标情绪类别训练
    python main.py --mode train --train_mode ppg_regression --target_emotion Stress
    python main.py --mode train --train_mode prv_regression --target_emotion Anxiety
    
    # 指定多个情绪类别训练
    python main.py --mode train --train_mode ppg_regression --emotions Stress,Anxiety,Sad

3. 情绪分类单任务训练（强制使用全部5种情绪数据）
--------------------------------------------------------------------------------
    # PPG情绪分类
    python main.py --mode train --train_mode ppg_classification
    
    # PRV情绪分类
    python main.py --mode train --train_mode prv_classification

4. 多任务训练（压力回归 + 情绪分类，强制使用全部5种情绪数据）
--------------------------------------------------------------------------------
    python main.py --mode train --train_mode multi_task

5. 基准模型训练（用于对比实验）
--------------------------------------------------------------------------------
    可选基准模型: lstm, gru, bilstm, tcn, transformer_baseline, informer
    
    # 基准模型压力回归
    python main.py --mode train --train_mode baseline_ppg_regression --baseline_model lstm
    python main.py --mode train --train_mode baseline_prv_regression --baseline_model gru
    python main.py --mode train --train_mode baseline_ppg_regression --baseline_model bilstm
    python main.py --mode train --train_mode baseline_prv_regression --baseline_model tcn
    
    # 基准模型情绪分类
    python main.py --mode train --train_mode baseline_ppg_classification --baseline_model transformer_baseline
    python main.py --mode train --train_mode baseline_prv_classification --baseline_model informer

6. 消融实验
--------------------------------------------------------------------------------
    # 自动运行所有消融配置
    python main.py --mode ablation
    
    # 手动禁用特定组件
    python main.py --mode train --train_mode dual_stream --no_physiological_pe
    python main.py --mode train --train_mode dual_stream --no_multi_scale_conv
    python main.py --mode train --train_mode dual_stream --no_freq_attention
    python main.py --mode train --train_mode dual_stream --no_stress_gating
    python main.py --mode train --train_mode dual_stream --no_cross_modal
    python main.py --mode train --train_mode dual_stream --no_uncertainty

7. 模型评估
--------------------------------------------------------------------------------
    python main.py --mode eval --model_path checkpoints/best_model.pth
    python main.py --mode eval --model_path checkpoints/best_model.pth --train_mode ppg_regression

8. 高级参数配置
--------------------------------------------------------------------------------
    # 指定配置文件
    python main.py --mode train --config default
    
    # 覆盖训练参数
    python main.py --mode train --train_mode dual_stream --epochs 100 --batch_size 32 --lr 0.001
    
    # 指定设备和随机种子
    python main.py --mode train --train_mode dual_stream --device cuda --seed 42
    
    # 自定义数据目录
    python main.py --mode train --train_mode dual_stream --ppg_dir /path/to/ppg --prv_dir /path/to/prv

================================================================================
训练模式说明
================================================================================

训练模式 (--train_mode):
    ppg_only / prv_only       : 单流模型训练
    dual_stream               : 双流融合模型训练
    ppg_regression            : PPG压力回归单任务
    prv_regression            : PRV压力回归单任务
    ppg_classification        : PPG情绪分类单任务
    prv_classification        : PRV情绪分类单任务
    multi_task                : 多任务训练（压力回归+情绪分类）
    baseline_ppg_regression   : 基准模型PPG压力回归
    baseline_prv_regression   : 基准模型PRV压力回归
    baseline_ppg_classification: 基准模型PPG情绪分类
    baseline_prv_classification: 基准模型PRV情绪分类

情绪数据使用规则:
    - 压力回归单任务: 可通过 --target_emotion 或 --emotions 指定情绪类别
    - 情绪分类单任务: 强制使用全部5种情绪，忽略情绪选择参数
    - 多任务训练: 强制使用全部5种情绪

可用情绪类别: Anxiety, Happy, Peace, Sad, Stress

可用基准模型 (--baseline_model):
    lstm, gru, bilstm, tcn, transformer_baseline, informer
"""

import os
import sys
import argparse
from datetime import datetime
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# 导入项目模块
from config import (
    get_config, list_available_configs, ExperimentConfig,
    TrainMode, TaskType, list_baseline_models, AVAILABLE_BASELINE_MODELS
)
from models import (
    PPGFormer, PRVModel, PPGFormerDualStream, DualStreamOnly,
    LSTMBaseline, TransformerBaseline, create_model, get_model_for_train_mode,
    create_baseline_model, BASELINE_MODELS, list_available_models,
    LSTMModel, GRUModel, BiLSTMModel, TCNModel, TransformerModel, InformerModel
)
from trainer import train_kfold, get_average_history, get_best_fold
from evaluator import Evaluator
from utils import (
    set_seed, Logger, EMOTION_NAMES, EMOTION_LABEL_MAP,
    load_emotion_data_from_folder, load_all_emotion_data,
    prepare_data_for_training, split_data_by_emotion,
    plot_training_process, plot_predictions, plot_fold_comparison,
    count_parameters, save_model, load_model, calculate_metrics
)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='PPG-Former-DualStream 训练与评估')
    
    parser.add_argument('--config', type=str, default='default',
                        choices=list_available_configs(),
                        help='配置名称')
    
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'eval', 'ablation'],
                        help='运行模式: train/eval/ablation')
    
    parser.add_argument('--train_mode', type=str, default=None,
                        choices=['ppg_only', 'prv_only', 'dual_stream',
                                'ppg_regression', 'prv_regression',
                                'ppg_classification', 'prv_classification',
                                'multi_task',
                                'baseline_ppg_regression', 'baseline_prv_regression',
                                'baseline_ppg_classification', 'baseline_prv_classification'],
                        help='训练模式')
    
    parser.add_argument('--baseline_model', type=str, default='lstm',
                        choices=['lstm', 'gru', 'bilstm', 'tcn', 'transformer_baseline', 'informer'],
                        help='基准模型名称')
    
    parser.add_argument('--task_type', type=str, default=None,
                        choices=['regression', 'classification', 'multi_task'],
                        help='任务类型')
    
    parser.add_argument('--model_path', type=str, default=None,
                        help='模型路径（评估模式使用）')
    
    parser.add_argument('--ppg_dir', type=str, default=None,
                        help='PPG数据目录')
    
    parser.add_argument('--prv_dir', type=str, default=None,
                        help='PRV数据目录')
    
    parser.add_argument('--emotions', type=str, default=None,
                        help='选择的情绪类别，逗号分隔 (如: Stress,Anxiety)')
    
    parser.add_argument('--target_emotion', type=str, default=None,
                        choices=['Anxiety', 'Happy', 'Peace', 'Sad', 'Stress'],
                        help='压力回归单任务的目标情绪类别（仅用于回归任务）')
    
    parser.add_argument('--epochs', type=int, default=None,
                        help='训练轮数（覆盖配置）')
    
    parser.add_argument('--batch_size', type=int, default=None,
                        help='批次大小（覆盖配置）')
    
    parser.add_argument('--lr', type=float, default=None,
                        help='学习率（覆盖配置）')
    
    parser.add_argument('--device', type=str, default=None,
                        help='设备: cuda/cpu')
    
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    
    # 消融实验参数
    parser.add_argument('--no_physiological_pe', action='store_true',
                        help='禁用生理周期位置编码')
    parser.add_argument('--no_multi_scale_conv', action='store_true',
                        help='禁用多尺度卷积')
    parser.add_argument('--no_freq_attention', action='store_true',
                        help='禁用频域注意力')
    parser.add_argument('--no_stress_gating', action='store_true',
                        help='禁用压力感知门控')
    parser.add_argument('--no_cross_modal', action='store_true',
                        help='禁用跨模态注意力')
    parser.add_argument('--no_uncertainty', action='store_true',
                        help='禁用不确定性加权')
    
    return parser.parse_args()


def load_data_new(config: ExperimentConfig, selected_emotions: List[str] = None):
    """
    加载新的数据结构
    
    根据任务类型自动选择情绪数据:
    - 压力回归单任务: 可以根据 target_emotion_for_regression 选择特定情绪类别的数据
    - 情绪分类单任务: 强制使用全部5种情绪类别的数据
    - 多任务: 强制使用全部5种情绪类别的数据
    """
    print("\n===== 加载数据 =====")
    
    ppg_dir = config.data.ppg_data_path
    prv_dir = config.data.prv_data_path
    train_mode = config.training.train_mode
    task_type = config.training.task_type
    
    # 检查数据目录
    ppg_exists = os.path.exists(ppg_dir)
    prv_exists = os.path.exists(prv_dir)
    
    if not ppg_exists and not prv_exists:
        print(f"错误: 数据目录不存在")
        print(f"  PPG: {ppg_dir}")
        print(f"  PRV: {prv_dir}")
        return None
    
    # 根据任务类型确定使用的情绪类别
    # 优先级: 函数参数 > 配置方法 > 默认值
    if selected_emotions is not None:
        # 如果显式传入了selected_emotions，但是分类任务必须使用全部情绪
        if task_type in ['classification', 'multi_task']:
            print(f"警告: 情绪分类/多任务必须使用全部情绪类别，忽略指定的情绪选择")
            emotions_to_use = config.data.get_emotions_for_task(task_type)
        else:
            # 压力回归任务可以使用指定的情绪
            emotions_to_use = selected_emotions
    else:
        # 使用配置方法自动获取情绪类别
        emotions_to_use = config.data.get_emotions_for_task(task_type)
    
    print(f"任务类型: {task_type}")
    print(f"使用情绪类别: {emotions_to_use}")
    
    # 对于分类任务，验证情绪数量
    if task_type in ['classification', 'multi_task'] and len(emotions_to_use) < 5:
        print(f"错误: 情绪分类/多任务必须使用全部5种情绪类别")
        return None
    
    # 替换原来的 selected_emotions 变量为 emotions_to_use
    selected_emotions = emotions_to_use
    
    # 根据训练模式决定加载哪些数据
    result = {}
    
    if train_mode in ['ppg_only', 'ppg_regression', 'ppg_classification',
                       'baseline_ppg_regression', 'baseline_ppg_classification']:
        # 仅加载PPG数据
        if not ppg_exists:
            print(f"错误: PPG数据目录不存在: {ppg_dir}")
            return None
        
        ppg_features, stress_labels, emotion_labels, stress_scaler = load_emotion_data_from_folder(
            ppg_dir, "PPG", selected_emotions, normalize_stress=True
        )
        
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        ppg_features = scaler.fit_transform(ppg_features)
        
        result['X_data'] = ppg_features
        result['y_stress'] = stress_labels
        result['y_emotion'] = emotion_labels
        result['X_prv'] = None
        result['scaler'] = scaler
        result['stress_scaler'] = stress_scaler  # 保存压力标签scaler用于反归一化
        
    elif train_mode in ['prv_only', 'prv_regression', 'prv_classification',
                         'baseline_prv_regression', 'baseline_prv_classification']:
        # 仅加载PRV数据
        if not prv_exists:
            print(f"错误: PRV数据目录不存在: {prv_dir}")
            return None
        
        prv_features, stress_labels, emotion_labels, stress_scaler = load_emotion_data_from_folder(
            prv_dir, "PRV", selected_emotions, normalize_stress=True
        )
        
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        prv_features = scaler.fit_transform(prv_features)
        
        result['X_data'] = prv_features
        result['y_stress'] = stress_labels
        result['y_emotion'] = emotion_labels
        result['X_prv'] = None
        result['scaler'] = scaler
        result['stress_scaler'] = stress_scaler  # 保存压力标签scaler用于反归一化
        
    else:
        # 加载双流数据 (dual_stream / multi_task)
        data = load_all_emotion_data(ppg_dir, prv_dir, selected_emotions, normalize=True, normalize_stress=True)
        
        result['X_data'] = data['ppg_features']
        result['X_prv'] = data.get('prv_features', None)
        result['y_stress'] = data['stress_labels']
        result['y_emotion'] = data['emotion_labels']
        result['scaler'] = data.get('ppg_scaler', None)
        result['stress_scaler'] = data.get('stress_scaler', None)  # 保存压力标签scaler用于反归一化
    
    # 验证分类任务必须使用全部5种情绪
    if task_type in ['classification', 'multi_task']:
        unique_emotions = len(np.unique(result['y_emotion']))
        if unique_emotions < 5:
            print(f"警告: 分类任务建议使用全部5种情绪数据，当前只有 {unique_emotions} 种")
    
    print(f"\n数据加载完成:")
    print(f"  主数据形状: {result['X_data'].shape}")
    if result['X_prv'] is not None:
        print(f"  PRV数据形状: {result['X_prv'].shape}")
    print(f"  压力标签形状: {result['y_stress'].shape}")
    print(f"  情绪标签分布: {dict(zip(*np.unique(result['y_emotion'], return_counts=True)))}")
    
    return result


def get_model_and_params(config: ExperimentConfig, has_prv: bool = False):
    """根据配置获取模型类和参数"""
    train_mode = config.training.train_mode
    task_type = config.training.task_type
    
    # 检查是否是基准模型训练模式
    if train_mode.startswith('baseline_'):
        return get_baseline_model_and_params(config)
    
    if train_mode in ['ppg_only', 'ppg_regression', 'ppg_classification']:
        model_class = PPGFormer
        model_params = {
            'input_dim': config.model.ppg_input_dim,
            'd_model': config.model.d_model,
            'n_heads': config.model.n_heads,
            'd_ff': config.model.d_ff,
            'num_layers': config.model.ppg_layers,
            'num_classes': config.model.num_emotions,
            'scales': config.model.scales,
            'dropout': config.model.dropout,
            'task_type': task_type,
            'use_physiological_pe': config.ablation.use_physiological_pe,
            'use_multi_scale_conv': config.ablation.use_multi_scale_conv,
            'use_time_freq_attention': config.ablation.use_time_freq_attention,
            'use_freq_attention': config.ablation.use_freq_attention,
            'use_stress_gating': config.ablation.use_stress_gating,
            'use_uncertainty_weighting': config.ablation.use_uncertainty_weighting
        }
        model_name = "PPGFormer"
        
    elif train_mode in ['prv_only', 'prv_regression', 'prv_classification']:
        model_class = PRVModel
        model_params = {
            'input_dim': config.model.prv_input_dim,
            'd_model': config.model.d_model,
            'n_heads': config.model.n_heads,
            'num_layers': config.model.prv_layers,
            'num_classes': config.model.num_emotions,
            'dropout': config.model.dropout,
            'task_type': task_type,
            'use_uncertainty_weighting': config.ablation.use_uncertainty_weighting
        }
        model_name = "PRVModel"
        
    else:
        # dual_stream / multi_task
        model_class = PPGFormerDualStream
        model_params = {
            'ppg_input_dim': config.model.ppg_input_dim,
            'prv_input_dim': config.model.prv_input_dim,
            'd_model': config.model.d_model,
            'n_heads': config.model.n_heads,
            'd_ff': config.model.d_ff,
            'ppg_layers': config.model.ppg_layers,
            'prv_layers': config.model.prv_layers,
            'fusion_layers': config.model.fusion_layers,
            'num_emotions': config.model.num_emotions,
            'scales': config.model.scales,
            'dropout': config.model.dropout,
            'use_physiological_pe': config.ablation.use_physiological_pe,
            'use_multi_scale_conv': config.ablation.use_multi_scale_conv,
            'use_time_freq_attention': config.ablation.use_time_freq_attention,
            'use_freq_attention': config.ablation.use_freq_attention,
            'use_stress_gating': config.ablation.use_stress_gating,
            'use_cross_modal_attention': config.ablation.use_cross_modal_attention,
            'use_uncertainty_weighting': config.ablation.use_uncertainty_weighting
        }
        model_name = "PPGFormerDualStream"
    
    return model_class, model_params, model_name


def get_baseline_model_and_params(config: ExperimentConfig):
    """获取基准模型类和参数"""
    train_mode = config.training.train_mode
    task_type = config.training.task_type
    baseline_config = config.baseline_model
    
    # 确定输入维度
    if 'ppg' in train_mode:
        input_dim = config.model.ppg_input_dim
    else:
        input_dim = config.model.prv_input_dim
    
    # 模型映射
    model_name_map = {
        'lstm': LSTMModel,
        'gru': GRUModel,
        'bilstm': BiLSTMModel,
        'tcn': TCNModel,
        'transformer_baseline': TransformerModel,
        'informer': InformerModel,
    }
    
    model_name = baseline_config.model_name.lower()
    if model_name not in model_name_map:
        raise ValueError(f"未知基准模型: {model_name}")
    
    model_class = model_name_map[model_name]
    
    # 构建模型参数
    model_params = {
        'input_dim': input_dim,
        'hidden_dim': baseline_config.hidden_dim,
        'num_layers': baseline_config.num_layers,
        'num_classes': baseline_config.num_classes,
        'dropout': baseline_config.dropout,
        'task_type': task_type,
    }
    
    # 为TCN添加kernel_size参数
    if model_name == 'tcn':
        model_params['kernel_size'] = baseline_config.kernel_size
    
    # 为Transformer/Informer添加n_heads参数
    if model_name in ['transformer_baseline', 'informer']:
        model_params['n_heads'] = baseline_config.n_heads
    
    # 模型显示名称
    display_name = f"Baseline_{model_name.upper()}"
    
    return model_class, model_params, display_name


def train_mode_func(args, config: ExperimentConfig):
    """训练模式"""
    # 解析选择的情绪
    selected_emotions = None
    if args.emotions:
        selected_emotions = [e.strip() for e in args.emotions.split(',')]
    
    # 加载数据
    data = load_data_new(config, selected_emotions)
    if data is None:
        return
    
    X_data = data['X_data']
    X_prv = data['X_prv']
    y_stress = data['y_stress']
    y_emotion = data['y_emotion']
    stress_scaler = data.get('stress_scaler', None)  # 获取压力标签scaler用于反归一化
    
    # 获取模型配置
    model_class, model_params, model_name = get_model_and_params(config, X_prv is not None)
    
    train_mode = config.training.train_mode
    task_type = config.training.task_type
    
    print(f"\n===== 训练配置 =====")
    print(f"训练模式: {train_mode}")
    print(f"任务类型: {task_type}")
    print(f"模型类型: {model_name}")
    print(f"模型参数:")
    for key, value in model_params.items():
        if not key.startswith('use_'):
            print(f"  {key}: {value}")
    
    # 创建日志器
    logger = Logger(
        log_dir=config.log.log_dir,
        experiment_name=f"{config.experiment_name}_{train_mode}",
        log_level=config.log.log_level
    )
    logger.log_config(config)
    
    # 设置设备
    device = args.device if args.device else config.device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        print("CUDA不可用，使用CPU")
    
    print(f"\n使用设备: {device}")
    
    # 创建临时模型查看参数量
    temp_model = model_class(**model_params)
    total_params, trainable_params = count_parameters(temp_model)
    print(f"模型总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    del temp_model
    
    # 确定是否使用情绪任务
    use_emotion = task_type in ['classification', 'multi_task']
    
    # 开始训练
    print(f"\n===== 开始 {config.training.k_folds} 折交叉验证训练 =====")
    
    all_histories, best_model_states, fold_results = train_kfold(
        model_class=model_class,
        model_params=model_params,
        X_data=X_data,
        X_prv=X_prv,
        y_stress=y_stress,
        y_emotion=y_emotion,
        config=config,
        logger=logger,
        use_emotion=use_emotion,
        stratify_by_emotion=(task_type == 'classification'),
        stress_scaler=stress_scaler  # 传递stress_scaler用于反归一化显示真实值
    )
    
    # 计算平均历史
    avg_history = get_average_history(all_histories)
    
    # 绘制训练过程
    if config.log.save_plots and task_type != 'classification':
        plot_training_process(
            avg_history['train_losses'], avg_history['val_losses'],
            avg_history['train_maes'], avg_history['val_maes'],
            avg_history['train_rmses'], avg_history['val_rmses'],
            model_name, config.log.result_dir, show=False
        )
        
        plot_fold_comparison(fold_results, model_name, config.log.result_dir, show=False)
    
    # 选择最佳模型
    best_fold_idx = get_best_fold(fold_results)
    print(f"\n选择第 {best_fold_idx + 1} 折的模型作为最佳模型")
    
    best_model = model_class(**model_params)
    best_model.load_state_dict(best_model_states[best_fold_idx])
    best_model.to(device)
    
    # 保存模型
    # model_save_path = os.path.join(
    #     config.log.model_save_dir,
    #     f"{model_name}_{train_mode}_{logger.timestamp}.pth"
    # )
    # save_model(best_model, model_save_path, config)
    
    # 保存日志
    logger.save_history()
    
    print("\n===== 训练完成 =====")
    print(f"最佳折结果:")
    for key, value in fold_results[best_fold_idx].items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
    print(f"日志已保存到: {logger.log_file}")
    
    # ===== 测试阶段：在测试集上评估并可视化 =====
    if task_type != 'classification':
        print("\n===== 测试集评估与可视化 =====")
        
        # 使用 train_test_split 直接划分测试集
        from sklearn.model_selection import train_test_split
        _, X_test, _, y_stress_test = train_test_split(
            X_data, y_stress,
            test_size=config.data.test_size,
            random_state=42,
            stratify=y_emotion
        )
        
        # 准备PRV测试数据（如果有）
        X_prv_test = None
        if X_prv is not None:
            _, X_prv_test = train_test_split(
                X_prv,
                test_size=config.data.test_size,
                random_state=42,
                stratify=y_emotion
            )
        
        # 创建测试数据加载器
        test_tensors = [torch.tensor(X_test, dtype=torch.float32)]
        if X_prv_test is not None and train_mode in ['dual_stream', 'multi_task']:
            test_tensors.append(torch.tensor(X_prv_test, dtype=torch.float32))
        test_tensors.append(torch.tensor(y_stress_test, dtype=torch.float32).reshape(-1, 1))
        
        test_dataset = TensorDataset(*test_tensors)
        test_loader = DataLoader(test_dataset, batch_size=config.training.batch_size, shuffle=False)
        
        # 使用最佳模型进行预测
        best_model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_data in test_loader:
                if len(batch_data) == 3:  # PPG + PRV + stress
                    ppg_data = batch_data[0].to(device)
                    prv_data = batch_data[1].to(device)
                    stress_target = batch_data[2]
                    
                    # 确保输入维度正确
                    if ppg_data.dim() == 2:
                        ppg_data = ppg_data.unsqueeze(-1)
                    if prv_data.dim() == 2:
                        prv_data = prv_data.unsqueeze(-1)
                    
                    # 前向传播
                    output = best_model(ppg_data, prv_data)
                    if isinstance(output, tuple):
                        output = output[0]
                else:  # PPG/PRV + stress
                    input_data = batch_data[0].to(device)
                    stress_target = batch_data[1]
                    
                    if input_data.dim() == 2:
                        input_data = input_data.unsqueeze(-1)
                    
                    output = best_model(input_data)
                    if isinstance(output, tuple):
                        output = output[0]
                
                # 收集预测和目标
                pred = output.squeeze().cpu().numpy()
                target = stress_target.squeeze().numpy()
                
                if np.ndim(pred) == 0:
                    all_predictions.append(pred.item())
                    all_targets.append(target.item())
                else:
                    all_predictions.extend(pred.tolist())
                    all_targets.extend(target.tolist())
        
        # 计算测试集评估指标
        predictions = np.array(all_predictions)
        targets = np.array(all_targets)
        
        # 反归一化压力标签，显示真实的评估指标
        if stress_scaler is not None:
            predictions_real = stress_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
            targets_real = stress_scaler.inverse_transform(targets.reshape(-1, 1)).flatten()
            print(f"\n测试集评估结果（反归一化后的真实压力值）:")
            real_metrics = calculate_metrics(predictions_real, targets_real)
            print(f"  MAE: {real_metrics['mae']:.4f}")
            print(f"  RMSE: {real_metrics['rmse']:.4f}")
            print(f"  R²: {real_metrics['r2']:.4f}")
            print(f"  相关系数: {real_metrics['correlation']:.4f}")
            print(f"  压力标签范围: [{targets_real.min():.2f}, {targets_real.max():.2f}]")
        else:
            predictions_real = predictions
            targets_real = targets
            real_metrics = calculate_metrics(predictions, targets)
            print(f"\n测试集评估结果:")
            print(f"  MAE: {real_metrics['mae']:.4f}")
            print(f"  RMSE: {real_metrics['rmse']:.4f}")
            print(f"  R²: {real_metrics['r2']:.4f}")
            print(f"  相关系数: {real_metrics['correlation']:.4f}")
        
        # 同时显示归一化后的指标（用于对比）
        norm_metrics = calculate_metrics(predictions, targets)
        print(f"\n测试集评估结果（归一化后 [0,1] 范围）:")
        print(f"  MAE: {norm_metrics['mae']:.4f}")
        print(f"  RMSE: {norm_metrics['rmse']:.4f}")
        print(f"  R²: {norm_metrics['r2']:.4f}")
        
        # 绘制预测散点图（使用真实压力值）
        save_path = plot_predictions(
            predictions_real.tolist(),
            targets_real.tolist(),
            model_name,
            config.log.result_dir,
            show=True
        )
        print(f"\n预测散点图已保存到: {save_path}")
        
        # 记录测试结果到日志（使用真实压力值指标）
        logger.log_test_result(
            test_loss=real_metrics['mse'],
            test_mae=real_metrics['mae'],
            test_rmse=real_metrics['rmse']
        )


def eval_mode_func(args, config: ExperimentConfig):
    """评估模式"""
    if args.model_path is None:
        print("错误: 评估模式需要指定模型路径 (--model_path)")
        return
    
    if not os.path.exists(args.model_path):
        print(f"错误: 模型文件不存在: {args.model_path}")
        return
    
    # 解析选择的情绪
    selected_emotions = None
    if args.emotions:
        selected_emotions = [e.strip() for e in args.emotions.split(',')]
    
    # 加载数据
    data = load_data_new(config, selected_emotions)
    if data is None:
        return
    
    X_data = data['X_data']
    X_prv = data['X_prv']
    y_stress = data['y_stress']
    y_emotion = data['y_emotion']
    stress_scaler = data.get('stress_scaler', None)  # 获取压力标签scaler用于反归一化
    
    # 获取模型配置
    model_class, model_params, model_name = get_model_and_params(config, X_prv is not None)
    
    train_mode = config.training.train_mode
    task_type = config.training.task_type
    
    # 设置设备
    device = args.device if args.device else config.device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    
    # 加载模型
    print(f"\n加载模型: {args.model_path}")
    model = model_class(**model_params)
    model = load_model(model, args.model_path, device)
    model.to(device)
    
    # 划分测试集
    split_result = split_data_by_emotion(
        X_data, y_stress, y_emotion,
        test_size=config.data.test_size, val_size=0
    )
    
    X_test = split_result['X_test']
    y_stress_test = split_result['y_stress_test']
    y_emotion_test = split_result['y_emotion_test']
    
    # 创建测试数据加载器
    if task_type == 'classification':
        test_tensors = [
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_emotion_test, dtype=torch.long)
        ]
    elif task_type == 'multi_task':
        if X_prv is not None:
            X_prv_test = split_result['X_test']  # 需要对应的PRV测试数据
            test_tensors = [
                torch.tensor(X_test, dtype=torch.float32),
                torch.tensor(X_prv_test, dtype=torch.float32),
                torch.tensor(y_stress_test, dtype=torch.float32).reshape(-1, 1),
                torch.tensor(y_emotion_test, dtype=torch.long)
            ]
        else:
            test_tensors = [
                torch.tensor(X_test, dtype=torch.float32),
                torch.tensor(y_stress_test, dtype=torch.float32).reshape(-1, 1),
                torch.tensor(y_emotion_test, dtype=torch.long)
            ]
    else:
        test_tensors = [
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_stress_test, dtype=torch.float32).reshape(-1, 1)
        ]
    
    test_dataset = TensorDataset(*test_tensors)
    test_loader = DataLoader(test_dataset, batch_size=config.training.batch_size)
    
    # 评估
    print("\n===== 模型评估 =====")
    evaluator = Evaluator(model, device, task_type=task_type, train_mode=train_mode)
    
    if task_type == 'classification':
        test_metrics = evaluator.evaluate_classification(test_loader)
    elif task_type == 'multi_task':
        test_metrics = evaluator.evaluate_multi_task(test_loader)
    else:
        test_metrics = evaluator.evaluate_regression(test_loader)
    
    # 绘制预测散点图（回归任务）
    if task_type != 'classification' and 'predictions' in test_metrics:
        print("\n===== 预测结果可视化 =====")
        
        # 打印详细评估指标
        print(f"测试集评估结果:")
        print(f"  MAE: {test_metrics['mae']:.4f}")
        print(f"  RMSE: {test_metrics['rmse']:.4f}")
        print(f"  R²: {test_metrics['r2']:.4f}")
        print(f"  相关系数: {test_metrics['correlation']:.4f}")
        
        # 绘制散点图
        save_path = plot_predictions(
            test_metrics['predictions'],
            test_metrics['targets'],
            model_name + "_eval",
            config.log.result_dir,
            show=True
        )
        print(f"\n预测散点图已保存到: {save_path}")


def ablation_mode_func(args, config: ExperimentConfig):
    """消融实验模式"""
    print("\n===== 消融实验 =====")
    
    # 加载数据
    data = load_data_new(config)
    if data is None:
        return
    
    X_data = data['X_data']
    X_prv = data['X_prv']
    y_stress = data['y_stress']
    y_emotion = data['y_emotion']
    
    # 设置设备
    device = args.device if args.device else config.device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    
    train_mode = config.training.train_mode
    task_type = config.training.task_type
    
    # 获取基础模型配置
    model_class, base_params, model_name = get_model_and_params(config, X_prv is not None)
    
    # 定义消融配置
    ablation_configs = {
        'Full Model (Baseline)': {},
        'w/o Physiological PE': {'use_physiological_pe': False},
        'w/o Multi-Scale Conv': {'use_multi_scale_conv': False},
        'w/o Freq Attention': {'use_freq_attention': False},
        'w/o Stress Gating': {'use_stress_gating': False},
    }
    
    # 如果是双流模型，添加更多消融配置
    if train_mode in ['dual_stream', 'multi_task']:
        ablation_configs['w/o Cross-Modal Attn'] = {'use_cross_modal_attention': False}
        ablation_configs['w/o Uncertainty Weighting'] = {'use_uncertainty_weighting': False}
    
    # 创建日志器
    logger = Logger(
        log_dir=config.log.log_dir,
        experiment_name=f"Ablation_{train_mode}",
        log_level=config.log.log_level
    )
    
    print(f"训练模式: {train_mode}")
    print(f"任务类型: {task_type}")
    print(f"消融配置数量: {len(ablation_configs)}")
    print()
    
    results = {}
    
    for config_name, changes in ablation_configs.items():
        print(f"\n{'='*50}")
        print(f"消融实验: {config_name}")
        print(f"{'='*50}")
        
        # 更新参数
        ablation_params = base_params.copy()
        ablation_params.update(changes)
        
        # 训练K折
        all_histories, best_model_states, fold_results = train_kfold(
            model_class=model_class,
            model_params=ablation_params,
            X_data=X_data,
            X_prv=X_prv,
            y_stress=y_stress,
            y_emotion=y_emotion,
            config=config,
            logger=None,  # 不记录到文件
            use_emotion=(task_type in ['classification', 'multi_task']),
            stratify_by_emotion=(task_type == 'classification')
        )
        
        # 计算平均结果
        avg_result = {
            'val_loss': np.mean([r['val_loss'] for r in fold_results]),
            'val_mae': np.mean([r['val_mae'] for r in fold_results]),
            'val_rmse': np.mean([r['val_rmse'] for r in fold_results]),
            'val_accuracy': np.mean([r.get('val_accuracy', 0) for r in fold_results]),
            'val_f1': np.mean([r.get('val_f1', 0) for r in fold_results]),
        }
        
        results[config_name] = avg_result
    
    # 打印结果表格
    print("\n" + "="*80)
    print("消融实验结果汇总")
    print("="*80)
    
    if task_type == 'classification':
        print(f"{'配置':<35} {'Accuracy':>12} {'F1':>12}")
        print("-" * 60)
        for name, metrics in results.items():
            print(f"{name:<35} {metrics['val_accuracy']:>12.4f} {metrics['val_f1']:>12.4f}")
    else:
        print(f"{'配置':<35} {'MAE':>12} {'RMSE':>12}")
        print("-" * 60)
        for name, metrics in results.items():
            print(f"{name:<35} {metrics['val_mae']:>12.4f} {metrics['val_rmse']:>12.4f}")
    
    # 保存结果
    logger.history['ablation_results'] = results
    logger.save_history()
    print(f"\n结果已保存到: {logger.log_file}")


def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 获取配置
    # 检查是否是基准模型训练模式
    if args.train_mode and args.train_mode.startswith('baseline_'):
        config = get_config(args.train_mode, args.baseline_model)
    else:
        config = get_config(args.config)
    
    # 覆盖配置
    if args.train_mode:
        config.training.train_mode = args.train_mode
        # 根据训练模式自动设置任务类型
        if args.train_mode in ['ppg_regression', 'prv_regression',
                                'baseline_ppg_regression', 'baseline_prv_regression']:
            config.training.task_type = 'regression'
        elif args.train_mode in ['ppg_classification', 'prv_classification',
                                   'baseline_ppg_classification', 'baseline_prv_classification']:
            config.training.task_type = 'classification'
        elif args.train_mode == 'multi_task':
            config.training.task_type = 'multi_task'
        
        # 如果是基准模型模式，设置基准模型名称
        if args.train_mode.startswith('baseline_'):
            config.use_baseline_model = True
            config.baseline_model.model_name = args.baseline_model
    
    if args.task_type:
        config.training.task_type = args.task_type
    
    if args.epochs:
        config.training.num_epochs = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.lr:
        config.training.learning_rate = args.lr
    if args.device:
        config.device = args.device
    if args.ppg_dir:
        config.data.data_root = os.path.dirname(args.ppg_dir)
        config.data.ppg_dir = os.path.basename(args.ppg_dir)
    if args.prv_dir:
        config.data.prv_dir = os.path.basename(args.prv_dir)
    
    # 处理目标情绪参数（仅用于压力回归任务）
    if args.target_emotion:
        if config.training.task_type == 'regression':
            config.data.target_emotion_for_regression = args.target_emotion
            print(f"压力回归任务目标情绪: {args.target_emotion}")
        else:
            print(f"警告: --target_emotion 参数仅对压力回归任务有效，当前任务类型为 {config.training.task_type}")
    
    # 应用消融实验参数
    if args.no_physiological_pe:
        config.ablation.use_physiological_pe = False
    if args.no_multi_scale_conv:
        config.ablation.use_multi_scale_conv = False
    if args.no_freq_attention:
        config.ablation.use_freq_attention = False
    if args.no_stress_gating:
        config.ablation.use_stress_gating = False
    if args.no_cross_modal:
        config.ablation.use_cross_modal_attention = False
    if args.no_uncertainty:
        config.ablation.use_uncertainty_weighting = False
    
    print("=" * 60)
    print("PPG-Former-DualStream 多任务压力预测模型")
    print("=" * 60)
    print(f"配置: {args.config}")
    print(f"模式: {args.mode}")
    print(f"训练模式: {config.training.train_mode}")
    print(f"任务类型: {config.training.task_type}")
    if config.use_baseline_model:
        print(f"基准模型: {config.baseline_model.model_name}")
    print(f"实验名称: {config.experiment_name}")
    
    # 执行对应模式
    if args.mode == 'train':
        train_mode_func(args, config)
    elif args.mode == 'eval':
        eval_mode_func(args, config)
    elif args.mode == 'ablation':
        ablation_mode_func(args, config)
    else:
        print(f"未知模式: {args.mode}")


if __name__ == "__main__":
    main()
