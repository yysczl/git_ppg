"""PRV 压力预测模型训练主程序

================================================================================
使用方法
================================================================================

1. 使用不同模型训练
--------------------------------------------------------------------------------
    # 使用LSTM模型（默认）
    python main.py --model_type lstm
    
    # 使用GRU模型
    python main.py --model_type gru
    
    # 使用BiLSTM模型
    python main.py --model_type bilstm
    
    # 使用TCN模型
    python main.py --model_type tcn
    
    # 使用Transformer模型
    python main.py --model_type transformer
    
    # 使用Informer模型
    python main.py --model_type informer

2. 使用预定义配置
--------------------------------------------------------------------------------
    python main.py --config lstm
    python main.py --config informer

3. 自定义参数
--------------------------------------------------------------------------------
    # 覆盖训练参数
    python main.py --model_type lstm --epochs 200 --batch_size 16 --lr 0.001
    
    # 指定数据文件
    python main.py --model_type lstm --data_file StressPRV.csv
    
    # 指定设备和随机种子
    python main.py --model_type lstm --device cuda --seed 42

4. 运行模式
--------------------------------------------------------------------------------
    # 训练模式（默认）
    python main.py --mode train --model_type lstm
    
    # 评估模式
    python main.py --mode eval --model_path checkpoints/prv_LSTM_model.pth

================================================================================
可用模型类型
================================================================================
    lstm, gru, bilstm, tcn, transformer, informer

"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

# 导入拆分后的模块
from data_processor import load_and_preprocess_data, set_seed
from train import evaluate_model, train_model_kfold, train_final_model, load_and_evaluate_model
from utils import plot_training_process, plot_predictions
from models import PRVDataset, LSTM, GRU, BiLSTM, TemporalConvNet, TimeSeriesTransformer, Informer
from config import (
    get_config, list_available_configs, list_available_models, 
    ExperimentConfig, AVAILABLE_MODELS
)


# 模型类映射
MODEL_CLASSES = {
    "lstm": LSTM,
    "gru": GRU,
    "bilstm": BiLSTM,
    "tcn": TemporalConvNet,
    "transformer": TimeSeriesTransformer,
    "informer": Informer,
}


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='PRV 压力预测模型训练与评估')
    
    parser.add_argument('--config', type=str, default='default',
                        choices=list_available_configs(),
                        help='配置名称')
    
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'eval'],
                        help='运行模式: train/eval')
    
    parser.add_argument('--model_type', type=str, default=None,
                        choices=AVAILABLE_MODELS,
                        help='模型类型: lstm/gru/bilstm/tcn/transformer/informer')
    
    parser.add_argument('--model_path', type=str, default=None,
                        help='模型路径（评估模式使用）')
    
    parser.add_argument('--data_file', type=str, default=None,
                        help='数据文件名')
    
    parser.add_argument('--data_root', type=str, default=None,
                        help='数据目录')
    
    parser.add_argument('--epochs', type=int, default=None,
                        help='训练轮数（覆盖配置）')
    
    parser.add_argument('--batch_size', type=int, default=None,
                        help='批次大小（覆盖配置）')
    
    parser.add_argument('--lr', type=float, default=None,
                        help='学习率（覆盖配置）')
    
    parser.add_argument('--k_folds', type=int, default=None,
                        help='K折交叉验证折数')
    
    parser.add_argument('--device', type=str, default=None,
                        help='设备: cuda/cpu')
    
    parser.add_argument('--seed', type=int, default=None,
                        help='随机种子')
    
    parser.add_argument('--signal_type', type=str, default='prv',
                        choices=['prv', 'ppg', 'pupil'],
                        help='信号类型')
    
    return parser.parse_args()


def get_model_class(model_type: str):
    """获取模型类"""
    model_type = model_type.lower()
    if model_type not in MODEL_CLASSES:
        raise ValueError(f"未知模型类型: {model_type}. 可用模型: {list(MODEL_CLASSES.keys())}")
    return MODEL_CLASSES[model_type]


def train_mode_func(args, config: ExperimentConfig):
    """训练模式"""
    # 获取数据路径
    file_path = config.data.data_path
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"文件 {file_path} 不存在!")
        return
    
    print(f"\n===== 数据加载 =====")
    print(f"数据文件: {file_path}")
    
    # 加载和预处理数据
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = load_and_preprocess_data(file_path)
    
    # 合并训练集和验证集用于k折交叉验证
    X_combined = np.concatenate([X_train, X_val], axis=0)
    y_combined = np.concatenate([y_train, y_val], axis=0)
    
    print("数据加载完成!")
    print(f"  训练+验证集大小: {X_combined.shape}")
    print(f"  测试集大小: {X_test.shape}")
    
    # 设置设备
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    # 获取模型类型和参数
    model_type = config.model_type.upper()
    model_class = get_model_class(config.model_type)
    model_params = config.get_model_config()
    
    print(f"\n===== 训练配置 =====")
    print(f"模型类型: {model_type}")
    print(f"实验名称: {config.experiment_name}")
    print(f"模型参数:")
    for key, value in model_params.items():
        print(f"  {key}: {value}")
    print(f"\n训练参数:")
    print(f"  Epochs: {config.training.num_epochs}")
    print(f"  Batch Size: {config.training.batch_size}")
    print(f"  Learning Rate: {config.training.learning_rate}")
    print(f"  K-Folds: {config.training.k_folds}")
    
    # 定义损失函数
    criterion = nn.MSELoss()
    
    # 定义优化器参数
    optimizer_class = optim.AdamW
    optimizer_params = {
        'lr': config.training.learning_rate,
        'weight_decay': config.training.weight_decay
    }
    
    # 定义学习率调度器参数
    scheduler_class = optim.lr_scheduler.ReduceLROnPlateau
    scheduler_params = {
        'mode': 'min',
        'factor': config.training.scheduler_factor,
        'patience': config.training.scheduler_patience
    }
    
    # K折交叉验证训练
    print(f"\n===== 开始 {config.training.k_folds} 折交叉验证训练 =====")
    
    fold_results, all_losses, best_model_states = train_model_kfold(
        X_combined, y_combined,
        model_class, model_params,
        criterion,
        optimizer_class, optimizer_params,
        config.training.num_epochs, device,
        config.training.k_folds, config.training.batch_size,
        scheduler_class, scheduler_params
    )
    
    # 绘制训练过程
    if config.log.save_plots:
        avg_train_losses = np.mean(all_losses[0], axis=0)
        avg_val_losses = np.mean(all_losses[1], axis=0)
        avg_train_maes = np.mean(all_losses[2], axis=0)
        avg_val_maes = np.mean(all_losses[3], axis=0)
        avg_train_rmses = np.mean(all_losses[4], axis=0)
        avg_val_rmses = np.mean(all_losses[5], axis=0)
        
        plot_training_process(
            avg_train_losses, avg_val_losses,
            avg_train_maes, avg_val_maes,
            avg_train_rmses, avg_val_rmses,
            model_type, config.data.signal_type
        )

    # 选择验证集表现最好的模型
    best_fold_index = np.argmin([result['val_loss'] for result in fold_results])
    print(f"\n选择第 {best_fold_index+1} 折的模型作为最佳模型")
    
    best_model = model_class(**model_params)
    best_model.load_state_dict(best_model_states[best_fold_index])
    best_model.to(device)
    
    # 创建测试数据加载器
    test_dataset = PRVDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=config.training.batch_size)
    
    # 评估模型
    print("\n===== 测试集评估 =====")
    test_loss, test_mae, test_rmse, predictions, targets = evaluate_model(
        best_model, test_loader, criterion, device
    )
    
    # 绘制预测结果
    if config.log.save_plots:
        plot_predictions(predictions, targets, model_type, config.data.signal_type)
    
    print("\n===== 交叉验证训练完成 =====")
    print(f"测试集 - Loss: {test_loss:.4f}, MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}")
    
    # 训练用于部署的最终模型
    print("\n===== 开始训练最终模型 =====")
    
    # 合并所有数据用于最终训练
    X_full = np.concatenate([X_train, X_val], axis=0)
    y_full = np.concatenate([y_train, y_val], axis=0)
    
    print(f"完整数据集大小: {X_full.shape}")
    
    # 使用与交叉验证相同的超参数训练最终模型
    final_model, final_train_losses, final_train_maes, final_train_rmses = train_final_model(
        X_full, y_full,
        model_class, model_params,
        criterion,
        optimizer_class, optimizer_params,
        config.training.num_epochs, device,
        config.training.batch_size,
        scheduler_class, scheduler_params
    )
    
    # 保存最终模型
    model_save_path = os.path.join(
        config.log.model_save_dir,
        f"{config.data.signal_type}_{model_type}_model.pth"
    )
    torch.save(final_model, model_save_path)
    print(f"\n最终模型已保存到: {model_save_path}")
    
    # 示例：加载保存的模型并重新评估测试集
    print("\n===== 加载模型并重新评估 =====")
    loaded_test_loss, loaded_test_mae, loaded_test_rmse, loaded_predictions, loaded_targets = load_and_evaluate_model(
        model_save_path, test_loader, criterion, device
    )
    
    print("\n加载模型评估结果:")
    print(f"测试集 - Loss: {loaded_test_loss:.4f}, MAE: {loaded_test_mae:.4f}, RMSE: {loaded_test_rmse:.4f}")
    
    print("\n===== 训练完成 =====")


def eval_mode_func(args, config: ExperimentConfig):
    """评估模式"""
    if args.model_path is None:
        print("错误: 评估模式需要指定模型路径 (--model_path)")
        return
    
    if not os.path.exists(args.model_path):
        print(f"错误: 模型文件不存在: {args.model_path}")
        return
    
    # 获取数据路径
    file_path = config.data.data_path
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"文件 {file_path} 不存在!")
        return
    
    print(f"\n===== 数据加载 =====")
    print(f"数据文件: {file_path}")
    
    # 加载和预处理数据
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = load_and_preprocess_data(file_path)
    
    print("数据加载完成!")
    print(f"  测试集大小: {X_test.shape}")
    
    # 设置设备
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    # 创建测试数据加载器
    test_dataset = PRVDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=config.training.batch_size)
    
    # 定义损失函数
    criterion = nn.MSELoss()
    
    # 加载并评估模型
    print(f"\n===== 加载模型: {args.model_path} =====")
    test_loss, test_mae, test_rmse, predictions, targets = load_and_evaluate_model(
        args.model_path, test_loader, criterion, device
    )
    
    print("\n===== 评估结果 =====")
    print(f"测试集 - Loss: {test_loss:.4f}, MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}")
    
    # 绘制预测结果
    if config.log.save_plots:
        model_type = config.model_type.upper()
        plot_predictions(predictions, targets, model_type + "_eval", config.data.signal_type)


def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    # 获取配置
    # 如果指定了model_type，使用对应的配置
    if args.model_type:
        config = get_config(args.model_type)
    else:
        config = get_config(args.config)
    
    # 覆盖配置参数
    if args.model_type:
        config.model_type = args.model_type
    if args.seed:
        config.training.seed = args.seed
    if args.epochs:
        config.training.num_epochs = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.lr:
        config.training.learning_rate = args.lr
    if args.k_folds:
        config.training.k_folds = args.k_folds
    if args.device:
        config.device = args.device
    if args.data_file:
        config.data.data_file = args.data_file
    if args.data_root:
        config.data.data_root = args.data_root
    if args.signal_type:
        config.data.signal_type = args.signal_type
    
    # 设置随机种子
    set_seed(config.training.seed)
    
    # 打印配置信息
    print("=" * 60)
    print("PRV 压力预测模型")
    print("=" * 60)
    print(f"配置: {args.config}")
    print(f"模式: {args.mode}")
    print(f"模型类型: {config.model_type.upper()}")
    print(f"信号类型: {config.data.signal_type}")
    print(f"实验名称: {config.experiment_name}")
    
    # 执行对应模式
    if args.mode == 'train':
        train_mode_func(args, config)
    elif args.mode == 'eval':
        eval_mode_func(args, config)
    else:
        print(f"未知模式: {args.mode}")


if __name__ == "__main__":
    main()
