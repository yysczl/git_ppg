"""
PPG-Former-DualStream 主程序入口
融合多尺度时频Transformer与双流协同的多任务心理压力预测

使用方法:
    python main.py                          # 使用默认配置训练
    python main.py --config ppg_only        # 仅使用PPG模型训练
    python main.py --config ablation_no_pe  # 消融实验：不使用生理周期位置编码
    python main.py --mode eval --model_path checkpoints/best_model.pth  # 评估模式
"""

import os
import sys
import argparse
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# 导入项目模块
from config import get_config, ExperimentConfig
from models import (
    PPGFormer, PPGFormerDualStream, DualStreamOnly,
    LSTMBaseline, TransformerBaseline, create_model
)
from trainer import train_kfold, get_average_history, get_best_fold
from evaluator import Evaluator, evaluate_model, compare_models
from utils import (
    set_seed, Logger,
    load_ppg_data, load_prv_data,
    plot_training_process, plot_predictions, plot_fold_comparison,
    count_parameters, save_model, load_model
)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='PPG-Former-DualStream 训练与评估')
    
    parser.add_argument('--config', type=str, default='default',
                        choices=['default', 'ppg_only', 'ablation_no_pe',
                                'ablation_no_freq', 'ablation_no_cross_attn',
                                'ablation_no_uncertainty'],
                        help='配置名称')
    
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'eval', 'ablation'],
                        help='运行模式: train/eval/ablation')
    
    parser.add_argument('--model_path', type=str, default=None,
                        help='模型路径（评估模式使用）')
    
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
    
    parser.add_argument('--use_emotion', action='store_true',
                        help='是否使用情绪分类任务')
    
    return parser.parse_args()


def load_data(config: ExperimentConfig):
    """加载数据"""
    print("\n===== 加载数据 =====")
    
    ppg_file = config.data.ppg_file_path
    prv_file = config.data.prv_file_path
    
    # 检查数据文件
    ppg_exists = os.path.exists(ppg_file)
    prv_exists = os.path.exists(prv_file)
    
    if not ppg_exists and not prv_exists:
        print(f"错误: 数据文件不存在")
        print(f"  PPG: {ppg_file}")
        print(f"  PRV: {prv_file}")
        return None, None, None, None, None, None
    
    # 加载PPG数据
    if ppg_exists:
        (X_ppg_train, y_ppg_train, X_ppg_val, y_ppg_val,
         X_ppg_test, y_ppg_test, ppg_scaler) = load_ppg_data(
            ppg_file, config.data.test_size, config.data.val_size
        )
        X_ppg_combined = np.concatenate([X_ppg_train, X_ppg_val], axis=0)
        y_combined = np.concatenate([y_ppg_train, y_ppg_val], axis=0)
    else:
        X_ppg_combined = None
        X_ppg_test = None
        y_ppg_test = None
    
    # 加载PRV数据
    if prv_exists:
        (X_prv_train, y_prv_train, X_prv_val, y_prv_val,
         X_prv_test, y_prv_test, prv_scaler) = load_prv_data(
            prv_file, config.data.test_size, config.data.val_size
        )
        X_prv_combined = np.concatenate([X_prv_train, X_prv_val], axis=0)
        
        if X_ppg_combined is None:
            X_ppg_combined = X_prv_combined
            X_ppg_test = X_prv_test
            y_combined = np.concatenate([y_prv_train, y_prv_val], axis=0)
            y_ppg_test = y_prv_test
    else:
        X_prv_combined = None
        X_prv_test = None
    
    print(f"\n数据加载完成:")
    if X_ppg_combined is not None:
        print(f"  PPG数据形状: {X_ppg_combined.shape}")
    if X_prv_combined is not None:
        print(f"  PRV数据形状: {X_prv_combined.shape}")
    print(f"  训练+验证集大小: {len(y_combined)}")
    print(f"  测试集大小: {len(y_ppg_test)}")
    
    return (X_ppg_combined, X_prv_combined, y_combined,
            X_ppg_test, X_prv_test, y_ppg_test)


def get_model_class_and_params(config: ExperimentConfig, X_prv=None):
    """根据配置获取模型类和参数"""
    
    if config.ablation.use_dual_stream and X_prv is not None:
        # 使用双流模型
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
    else:
        # 使用单流PPG-Former模型
        model_class = PPGFormer
        model_params = {
            'input_dim': config.model.ppg_input_dim,
            'output_dim': 1,
            'd_model': config.model.d_model,
            'n_heads': config.model.n_heads,
            'd_ff': config.model.d_ff,
            'num_layers': config.model.ppg_layers,
            'scales': config.model.scales,
            'dropout': config.model.dropout,
            'use_physiological_pe': config.ablation.use_physiological_pe,
            'use_multi_scale_conv': config.ablation.use_multi_scale_conv,
            'use_time_freq_attention': config.ablation.use_time_freq_attention,
            'use_freq_attention': config.ablation.use_freq_attention,
            'use_stress_gating': config.ablation.use_stress_gating
        }
        model_name = "PPGFormer"
    
    return model_class, model_params, model_name


def train_mode(args, config: ExperimentConfig):
    """训练模式"""
    # 加载数据
    data = load_data(config)
    if data[0] is None:
        return
    
    X_ppg, X_prv, y_stress, X_ppg_test, X_prv_test, y_test = data
    
    # 获取模型配置
    model_class, model_params, model_name = get_model_class_and_params(config, X_prv)
    
    print(f"\n===== 模型配置 =====")
    print(f"模型类型: {model_name}")
    print(f"模型参数:")
    for key, value in model_params.items():
        print(f"  {key}: {value}")
    
    # 创建日志器
    logger = Logger(
        log_dir=config.log.log_dir,
        experiment_name=config.experiment_name,
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
    
    # 开始训练
    print(f"\n===== 开始 {config.training.k_folds} 折交叉验证训练 =====")
    
    all_histories, best_model_states, fold_results = train_kfold(
        model_class=model_class,
        model_params=model_params,
        X_ppg=X_ppg,
        X_prv=X_prv,
        y_stress=y_stress,
        y_emotion=None,
        config=config,
        logger=logger,
        use_emotion=args.use_emotion
    )
    
    # 计算平均历史
    avg_history = get_average_history(all_histories)
    
    # 绘制训练过程
    if config.log.save_plots:
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
    
    # 创建测试数据加载器
    if X_prv_test is not None:
        test_dataset = TensorDataset(
            torch.tensor(X_ppg_test, dtype=torch.float32),
            torch.tensor(X_prv_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)
        )
    else:
        test_dataset = TensorDataset(
            torch.tensor(X_ppg_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)
        )
    test_loader = DataLoader(test_dataset, batch_size=config.training.batch_size)
    
    # 评估
    print("\n===== 在测试集上评估 =====")
    evaluator = Evaluator(best_model, device, logger)
    test_metrics = evaluator.evaluate_regression(test_loader)
    
    # 绘制预测结果
    if config.log.save_plots:
        plot_predictions(
            test_metrics['predictions'], test_metrics['targets'],
            model_name, config.log.result_dir, show=False
        )
    
    # 记录测试结果
    logger.log_test_result(test_metrics['loss'], test_metrics['mae'], test_metrics['rmse'])
    
    # 保存模型
    model_save_path = os.path.join(
        config.log.model_save_dir,
        f"{model_name}_{logger.timestamp}.pth"
    )
    save_model(best_model, model_save_path, config)
    
    # 保存日志
    logger.save_history()
    
    print("\n===== 训练完成 =====")
    print(f"最终测试结果:")
    print(f"  Loss: {test_metrics['loss']:.4f}")
    print(f"  MAE: {test_metrics['mae']:.4f}")
    print(f"  RMSE: {test_metrics['rmse']:.4f}")
    print(f"  R²: {test_metrics['r2']:.4f}")
    print(f"\n模型已保存到: {model_save_path}")
    print(f"日志已保存到: {logger.log_file}")


def eval_mode(args, config: ExperimentConfig):
    """评估模式"""
    if args.model_path is None:
        print("错误: 评估模式需要指定模型路径 (--model_path)")
        return
    
    if not os.path.exists(args.model_path):
        print(f"错误: 模型文件不存在: {args.model_path}")
        return
    
    # 加载数据
    data = load_data(config)
    if data[0] is None:
        return
    
    X_ppg, X_prv, y_stress, X_ppg_test, X_prv_test, y_test = data
    
    # 获取模型配置
    model_class, model_params, model_name = get_model_class_and_params(config, X_prv)
    
    # 设置设备
    device = args.device if args.device else config.device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    
    # 加载模型
    print(f"\n加载模型: {args.model_path}")
    model = model_class(**model_params)
    model = load_model(model, args.model_path, device)
    model.to(device)
    
    # 创建测试数据加载器
    if X_prv_test is not None:
        test_dataset = TensorDataset(
            torch.tensor(X_ppg_test, dtype=torch.float32),
            torch.tensor(X_prv_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)
        )
    else:
        test_dataset = TensorDataset(
            torch.tensor(X_ppg_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)
        )
    test_loader = DataLoader(test_dataset, batch_size=config.training.batch_size)
    
    # 评估
    print("\n===== 模型评估 =====")
    evaluator = Evaluator(model, device)
    test_metrics = evaluator.evaluate_regression(test_loader)
    
    # 绘制结果
    if config.log.save_plots:
        plot_predictions(
            test_metrics['predictions'], test_metrics['targets'],
            model_name + "_eval", config.log.result_dir, show=True
        )


def ablation_mode(args, config: ExperimentConfig):
    """消融实验模式"""
    print("\n===== 消融实验 =====")
    
    # 加载数据
    data = load_data(config)
    if data[0] is None:
        return
    
    X_ppg, X_prv, y_stress, X_ppg_test, X_prv_test, y_test = data
    
    # 设置设备
    device = args.device if args.device else config.device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    
    # 定义消融配置
    ablation_configs = {
        'w/o Physiological PE': {
            'use_physiological_pe': False
        },
        'w/o Multi-Scale Conv': {
            'use_multi_scale_conv': False
        },
        'w/o Freq Attention': {
            'use_freq_attention': False
        },
        'w/o Stress Gating': {
            'use_stress_gating': False
        },
        'w/o Cross-Modal Attn': {
            'use_cross_modal_attention': False
        },
        'w/o Uncertainty Weighting': {
            'use_uncertainty_weighting': False
        }
    }
    
    # 基础模型参数
    if X_prv is not None:
        base_params = {
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
            'use_physiological_pe': True,
            'use_multi_scale_conv': True,
            'use_time_freq_attention': True,
            'use_freq_attention': True,
            'use_stress_gating': True,
            'use_cross_modal_attention': True,
            'use_uncertainty_weighting': True
        }
        model_class = PPGFormerDualStream
    else:
        base_params = {
            'input_dim': config.model.ppg_input_dim,
            'output_dim': 1,
            'd_model': config.model.d_model,
            'n_heads': config.model.n_heads,
            'd_ff': config.model.d_ff,
            'num_layers': config.model.ppg_layers,
            'scales': config.model.scales,
            'dropout': config.model.dropout,
            'use_physiological_pe': True,
            'use_multi_scale_conv': True,
            'use_time_freq_attention': True,
            'use_freq_attention': True,
            'use_stress_gating': True
        }
        model_class = PPGFormer
        # 移除双流相关的消融配置
        del ablation_configs['w/o Cross-Modal Attn']
        del ablation_configs['w/o Uncertainty Weighting']
    
    # 创建测试数据加载器
    if X_prv_test is not None:
        test_dataset = TensorDataset(
            torch.tensor(X_ppg_test, dtype=torch.float32),
            torch.tensor(X_prv_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)
        )
    else:
        test_dataset = TensorDataset(
            torch.tensor(X_ppg_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)
        )
    test_loader = DataLoader(test_dataset, batch_size=config.training.batch_size)
    
    print("\n注意: 消融实验需要先训练每个配置的模型")
    print("这里仅展示未训练模型的初始评估结果（用于验证代码正确性）")
    print("实际消融实验需要对每个配置进行完整训练\n")
    
    # 评估各配置
    results = {}
    
    # 基线模型
    print("评估: Baseline (Full Model)")
    baseline_model = model_class(**base_params)
    evaluator = Evaluator(baseline_model, device)
    baseline_metrics = evaluator.evaluate_regression(test_loader, return_predictions=False)
    results['Baseline'] = baseline_metrics
    
    # 消融配置
    for config_name, changes in ablation_configs.items():
        print(f"\n评估: {config_name}")
        
        ablation_params = base_params.copy()
        ablation_params.update(changes)
        
        ablation_model = model_class(**ablation_params)
        evaluator = Evaluator(ablation_model, device)
        metrics = evaluator.evaluate_regression(test_loader, return_predictions=False)
        results[config_name] = metrics
    
    # 打印结果表格
    print("\n===== 消融实验结果汇总 =====")
    print(f"{'配置':<30} {'RMSE':>10} {'MAE':>10} {'R²':>10}")
    print("-" * 60)
    
    for name, metrics in results.items():
        print(f"{name:<30} {metrics['rmse']:>10.4f} {metrics['mae']:>10.4f} {metrics['r2']:>10.4f}")


def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 获取配置
    config = get_config(args.config)
    
    # 覆盖配置
    if args.epochs:
        config.training.num_epochs = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.lr:
        config.training.learning_rate = args.lr
    if args.device:
        config.device = args.device
    
    print("=" * 60)
    print("PPG-Former-DualStream 多任务压力预测模型")
    print("=" * 60)
    print(f"配置: {args.config}")
    print(f"模式: {args.mode}")
    print(f"实验名称: {config.experiment_name}")
    
    # 执行对应模式
    if args.mode == 'train':
        train_mode(args, config)
    elif args.mode == 'eval':
        eval_mode(args, config)
    elif args.mode == 'ablation':
        ablation_mode(args, config)
    else:
        print(f"未知模式: {args.mode}")


if __name__ == "__main__":
    main()
