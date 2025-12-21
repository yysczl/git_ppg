"""测试训练脚本 - 使用 informer-prv/models.py 中的基准模型
使用方法:
    # 使用 LSTM 训练 PPG 压力回归
    python test_main.py --model lstm --data_type ppg --task regression
    
    # 使用 GRU 训练 PRV 压力回归
    python test_main.py --model gru --data_type prv --task regression
    
    # 使用 BiLSTM 训练 PPG 情绪分类
    python test_main.py --model bilstm --data_type ppg --task classification
    
    # 使用 Transformer 训练
    python test_main.py --model transformer --data_type ppg --task regression
    
    # 使用 Informer 训练
    python test_main.py --model informer --data_type ppg --task regression
    
    # 使用 TCN 训练
    python test_main.py --model tcn --data_type ppg --task regression

可用模型: lstm, gru, bilstm, tcn, transformer, informer
数据类型: ppg, prv
任务类型: regression, classification
"""

import os
import sys
import argparse
from datetime import datetime
from typing import List, Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from basemodel import (
    LSTM, GRU, BiLSTM, 
    TemporalConvNet, TimeSeriesTransformer, Informer
)

# 从本地 utils 导入工具函数
from utils import (
    set_seed, Logger, EMOTION_NAMES, EMOTION_LABEL_MAP,
    load_emotion_data_from_folder, calculate_metrics, calculate_classification_metrics,
    plot_training_process, plot_predictions, plot_fold_comparison
)


# 模型映射字典
MODEL_MAP = {
    'lstm': LSTM,
    'gru': GRU,
    'bilstm': BiLSTM,
    'tcn': TemporalConvNet,
    'transformer': TimeSeriesTransformer,
    'informer': Informer,
}


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='基准模型测试训练')
    
    parser.add_argument('--model', type=str, default='lstm',
                        choices=list(MODEL_MAP.keys()),
                        help='模型类型')
    
    parser.add_argument('--data_type', type=str, default='ppg',
                        choices=['ppg', 'prv'],
                        help='数据类型: ppg/prv')
    
    parser.add_argument('--task', type=str, default='regression',
                        choices=['regression', 'classification'],
                        help='任务类型: regression/classification')
    
    parser.add_argument('--data_dir', type=str, default='dataset',
                        help='数据目录路径')
    
    parser.add_argument('--emotions', type=str, default=None,
                        help='选择的情绪类别，逗号分隔 (如: Stress,Anxiety)')
    
    parser.add_argument('--epochs', type=int, default=300,
                        help='训练轮数')
    
    parser.add_argument('--batch_size', type=int, default=8,
                        help='批次大小')
    
    parser.add_argument('--lr', type=float, default=0.005,
                        help='学习率')
    
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='隐藏层维度')
    
    parser.add_argument('--num_layers', type=int, default=2,
                        help='模型层数')
    
    parser.add_argument('--k_folds', type=int, default=5,
                        help='K折交叉验证折数')
    
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备: cuda/cpu')
    
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    
    return parser.parse_args()


def create_model(model_name: str, input_dim: int, output_dim: int, 
                 hidden_dim: int, num_layers: int, task_type: str):
    """
    创建模型
    
    Args:
        model_name: 模型名称
        input_dim: 输入维度
        output_dim: 输出维度 (分类任务为类别数，回归任务为1)
        hidden_dim: 隐藏层维度
        num_layers: 模型层数
        task_type: 任务类型
    
    Returns:
        model: 创建的模型
    """
    model_class = MODEL_MAP.get(model_name.lower())
    
    if model_class is None:
        raise ValueError(f"未知模型: {model_name}. 可用模型: {list(MODEL_MAP.keys())}")
    
    if model_name == 'lstm':
        model = LSTM(input_dim, hidden_dim, num_layers, output_dim)
    elif model_name == 'gru':
        model = GRU(input_dim, hidden_dim, num_layers, output_dim)
    elif model_name == 'bilstm':
        model = BiLSTM(input_dim, hidden_dim, num_layers, output_dim)
    elif model_name == 'tcn':
        # TCN 需要 num_channels 参数
        num_channels = [hidden_dim] * num_layers
        model = TemporalConvNet(input_dim, output_dim, num_channels, kernel_size=3, dropout=0.1)
    elif model_name == 'transformer':
        model = TimeSeriesTransformer(
            input_dim=input_dim,
            num_heads=8,
            num_layers=num_layers,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            dropout_rate=0.1
        )
    elif model_name == 'informer':
        model = Informer(
            input_dim=input_dim,
            output_dim=output_dim,
            d_model=hidden_dim,
            n_heads=8,
            d_ff=hidden_dim * 2,
            enc_layers=num_layers,
            dropout=0.1
        )
    else:
        raise ValueError(f"未实现模型: {model_name}")
    
    return model


def load_data(data_dir: str, data_type: str, selected_emotions: List[str] = None):
    """
    加载数据
    
    Args:
        data_dir: 数据目录
        data_type: 数据类型 (ppg/prv)
        selected_emotions: 选择的情绪类别
    
    Returns:
        features, stress_labels, emotion_labels
    """
    if data_type == 'ppg':
        folder = os.path.join(data_dir, 'PPG')
        signal_type = 'PPG'
    else:
        folder = os.path.join(data_dir, 'PRV')
        signal_type = 'PRV'
    
    if not os.path.exists(folder):
        raise FileNotFoundError(f"数据目录不存在: {folder}")
    
    features, stress_labels, emotion_labels = load_emotion_data_from_folder(
        folder, signal_type, selected_emotions
    )
    
    # 标准化
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    
    return features, stress_labels, emotion_labels, scaler


def train_epoch(model, train_loader, criterion, optimizer, device, task_type):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    total_mae = 0
    total_correct = 0
    total_samples = 0
    batch_count = 0
    
    for batch_data in train_loader:
        if task_type == 'classification':
            features, labels = batch_data
            features = features.to(device)
            labels = labels.to(device)
        else:
            features, labels = batch_data
            features = features.to(device)
            labels = labels.to(device)
        
        # 确保输入维度正确 [batch, seq_len, input_dim]
        if features.dim() == 2:
            features = features.unsqueeze(-1)
        
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(features)
        outputs = outputs.squeeze()
        
        # 计算损失
        if task_type == 'classification':
            loss = criterion(outputs, labels)
            # 计算准确率
            pred = torch.argmax(outputs, dim=1)
            total_correct += (pred == labels).sum().item()
            total_samples += labels.size(0)
        else:
            loss = criterion(outputs, labels.squeeze())
            # 计算 MAE
            mae = torch.abs(outputs - labels.squeeze()).mean().item()
            total_mae += mae
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        batch_count += 1
    
    result = {'loss': total_loss / batch_count}
    
    if task_type == 'classification':
        result['accuracy'] = total_correct / total_samples if total_samples > 0 else 0
    else:
        result['mae'] = total_mae / batch_count
    
    return result


def validate_epoch(model, val_loader, criterion, device, task_type):
    """验证一个 epoch"""
    model.eval()
    total_loss = 0
    total_mae = 0
    total_rmse = 0
    total_correct = 0
    total_samples = 0
    batch_count = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_data in val_loader:
            if task_type == 'classification':
                features, labels = batch_data
                features = features.to(device)
                labels = labels.to(device)
            else:
                features, labels = batch_data
                features = features.to(device)
                labels = labels.to(device)
            
            # 确保输入维度正确
            if features.dim() == 2:
                features = features.unsqueeze(-1)
            
            # 前向传播
            outputs = model(features)
            outputs = outputs.squeeze()
            
            # 计算损失
            if task_type == 'classification':
                loss = criterion(outputs, labels)
                pred = torch.argmax(outputs, dim=1)
                total_correct += (pred == labels).sum().item()
                total_samples += labels.size(0)
                all_preds.extend(pred.cpu().numpy().tolist())
                all_targets.extend(labels.cpu().numpy().tolist())
            else:
                loss = criterion(outputs, labels.squeeze())
                diff = torch.abs(outputs - labels.squeeze())
                total_mae += diff.mean().item()
                total_rmse += torch.sqrt((diff ** 2).mean()).item()
                all_preds.extend(outputs.cpu().numpy().tolist())
                all_targets.extend(labels.cpu().numpy().squeeze().tolist())
            
            total_loss += loss.item()
            batch_count += 1
    
    result = {'loss': total_loss / batch_count}
    
    if task_type == 'classification':
        result['accuracy'] = total_correct / total_samples if total_samples > 0 else 0
        if len(all_preds) > 0:
            cls_metrics = calculate_classification_metrics(
                np.array(all_preds), np.array(all_targets)
            )
            result['f1'] = cls_metrics['f1_macro']
    else:
        result['mae'] = total_mae / batch_count
        result['rmse'] = total_rmse / batch_count
    
    result['predictions'] = all_preds
    result['targets'] = all_targets
    
    return result


def train_single_fold(model, train_loader, val_loader, 
                      num_epochs, lr, device, task_type, print_freq=10,
                      weight_decay=1e-4, scheduler_patience=20):
    """
    训练单折模型
    
    Args:
        model: 模型实例
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        num_epochs: 训练轮数
        lr: 学习率
        device: 设备
        task_type: 任务类型 (regression/classification)
        print_freq: 打印频率
        weight_decay: 权重衰减
        scheduler_patience: 学习率调度器耐心值
    
    Returns:
        history, best_model_state, best_val_result
    """
    # 损失函数
    if task_type == 'classification':
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()
    
    # 优化器（与 prv_informer.py 一致）
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # 学习率调度器（与 prv_informer.py 一致）
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=scheduler_patience
    )
    
    history = {
        'train_losses': [], 'val_losses': [],
        'train_maes': [], 'val_maes': [],
        'train_rmses': [], 'val_rmses': [],
        'train_accs': [], 'val_accs': [],
        'val_f1s': [],
    }
    
    best_val_loss = float('inf')
    best_model_state = None
    best_val_result = None
    
    model.to(device)
    
    for epoch in range(num_epochs):
        # 训练
        train_result = train_epoch(model, train_loader, criterion, optimizer, device, task_type)
        
        # 验证
        val_result = validate_epoch(model, val_loader, criterion, device, task_type)
        
        # 更新学习率
        scheduler.step(val_result['loss'])
        
        # 保存最佳模型
        if val_result['loss'] < best_val_loss:
            best_val_loss = val_result['loss']
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_val_result = val_result.copy()
        
        # 记录历史
        history['train_losses'].append(train_result['loss'])
        history['val_losses'].append(val_result['loss'])
        history['train_maes'].append(train_result.get('mae', 0))
        history['val_maes'].append(val_result.get('mae', 0))
        history['val_rmses'].append(val_result.get('rmse', 0))
        history['train_accs'].append(train_result.get('accuracy', 0))
        history['val_accs'].append(val_result.get('accuracy', 0))
        history['val_f1s'].append(val_result.get('f1', 0))
        
        # 打印进度
        if (epoch + 1) % print_freq == 0:
            current_lr = optimizer.param_groups[0]['lr']
            if task_type == 'classification':
                print(f'  Epoch {epoch+1}/{num_epochs}: '
                      f'Train Loss={train_result["loss"]:.4f}, Acc={train_result.get("accuracy", 0):.4f}, '
                      f'Val Loss={val_result["loss"]:.4f}, Acc={val_result.get("accuracy", 0):.4f}, '
                      f'LR={current_lr:.6f}')
            else:
                print(f'  Epoch {epoch+1}/{num_epochs}: '
                      f'Train Loss={train_result["loss"]:.4f}, MAE={train_result.get("mae", 0):.4f}, '
                      f'Val Loss={val_result["loss"]:.4f}, MAE={val_result.get("mae", 0):.4f}, '
                      f'RMSE={val_result.get("rmse", 0):.4f}, LR={current_lr:.6f}')
    
    return history, best_model_state, best_val_result


def train_kfold(args, features, stress_labels, emotion_labels):
    """
    K折交叉验证训练（与 prv_informer.py 流程一致）
    
    Returns:
        fold_results: 每折的结果
        all_histories: 所有折的训练历史
        best_model_states: 所有折的最佳模型状态
        device: 设备
        input_dim: 输入维度
        output_dim: 输出维度
    """
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    # 确定输入维度和输出维度
    if args.data_type == 'ppg':
        input_dim = 1
        seq_len = 1800
    else:
        input_dim = 1
        seq_len = 80
    
    if args.task == 'classification':
        output_dim = len(EMOTION_NAMES)
    else:
        output_dim = 1
    
    print(f"输入维度: {input_dim}, 序列长度: {seq_len}, 输出维度: {output_dim}")
    
    # 准备K折
    k_folds = args.k_folds
    
    if args.task == 'classification':
        kfold = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=args.seed)
        split_data = emotion_labels
    else:
        kfold = KFold(n_splits=k_folds, shuffle=True, random_state=args.seed)
        split_data = features
    
    all_fold_results = []
    all_histories = []  # 收集所有折的训练历史
    best_model_states = []  # 收集所有折的最佳模型状态
    
    print(f"\n开始 {k_folds} 折交叉验证训练...")
    
    for fold, (train_ids, val_ids) in enumerate(kfold.split(features, split_data if args.task == 'classification' else None)):
        fold_num = fold + 1
        print(f"\n{'='*50}")
        print(f"第 {fold_num}/{k_folds} 折")
        print(f"训练集大小: {len(train_ids)}, 验证集大小: {len(val_ids)}")
        print(f"{'='*50}")
        
        # 划分数据
        X_train, X_val = features[train_ids], features[val_ids]
        
        if args.task == 'classification':
            y_train = emotion_labels[train_ids]
            y_val = emotion_labels[val_ids]
            train_tensors = [
                torch.tensor(X_train, dtype=torch.float32),
                torch.tensor(y_train, dtype=torch.long)
            ]
            val_tensors = [
                torch.tensor(X_val, dtype=torch.float32),
                torch.tensor(y_val, dtype=torch.long)
            ]
        else:
            y_train = stress_labels[train_ids]
            y_val = stress_labels[val_ids]
            train_tensors = [
                torch.tensor(X_train, dtype=torch.float32),
                torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
            ]
            val_tensors = [
                torch.tensor(X_val, dtype=torch.float32),
                torch.tensor(y_val, dtype=torch.float32).reshape(-1, 1)
            ]
        
        # 创建数据加载器
        train_dataset = TensorDataset(*train_tensors)
        val_dataset = TensorDataset(*val_tensors)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
        
        # 创建模型
        model = create_model(
            args.model, input_dim, output_dim,
            args.hidden_dim, args.num_layers, args.task
        )
        
        # 打印模型参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"模型参数量: {total_params:,} (可训练: {trainable_params:,})")
        
        # 训练
        history, best_state, best_result = train_single_fold(
            model, train_loader, val_loader,
            args.epochs, args.lr, device, args.task
        )
        
        # 收集训练历史和最佳模型状态
        all_histories.append(history)
        best_model_states.append(best_state)
        
        # 记录结果
        fold_result = {
            'fold': fold_num,
            'val_loss': best_result['loss'],
            'val_mae': best_result.get('mae', 0),
            'val_rmse': best_result.get('rmse', 0),
            'val_accuracy': best_result.get('accuracy', 0),
            'val_f1': best_result.get('f1', 0),
        }
        all_fold_results.append(fold_result)
        
        # 打印该折结果
        if args.task == 'classification':
            print(f"\n第 {fold_num} 折最佳结果: "
                  f"Val Loss={fold_result['val_loss']:.4f}, "
                  f"Acc={fold_result['val_accuracy']:.4f}, "
                  f"F1={fold_result['val_f1']:.4f}")
        else:
            print(f"\n第 {fold_num} 折最佳结果: "
                  f"Val Loss={fold_result['val_loss']:.4f}, "
                  f"MAE={fold_result['val_mae']:.4f}, "
                  f"RMSE={fold_result['val_rmse']:.4f}")
    
    # 计算平均结果
    print(f"\n{'='*60}")
    print("K折交叉验证结果汇总")
    print(f"{'='*60}")
    
    avg_loss = np.mean([r['val_loss'] for r in all_fold_results])
    std_loss = np.std([r['val_loss'] for r in all_fold_results])
    
    if args.task == 'classification':
        avg_acc = np.mean([r['val_accuracy'] for r in all_fold_results])
        std_acc = np.std([r['val_accuracy'] for r in all_fold_results])
        avg_f1 = np.mean([r['val_f1'] for r in all_fold_results])
        std_f1 = np.std([r['val_f1'] for r in all_fold_results])
        print(f"平均 Val Loss: {avg_loss:.4f} ± {std_loss:.4f}")
        print(f"平均 Accuracy: {avg_acc:.4f} ± {std_acc:.4f}")
        print(f"平均 F1 Score: {avg_f1:.4f} ± {std_f1:.4f}")
    else:
        avg_mae = np.mean([r['val_mae'] for r in all_fold_results])
        std_mae = np.std([r['val_mae'] for r in all_fold_results])
        avg_rmse = np.mean([r['val_rmse'] for r in all_fold_results])
        std_rmse = np.std([r['val_rmse'] for r in all_fold_results])
        print(f"平均 Val Loss: {avg_loss:.4f} ± {std_loss:.4f}")
        print(f"平均 MAE: {avg_mae:.4f} ± {std_mae:.4f}")
        print(f"平均 RMSE: {avg_rmse:.4f} ± {std_rmse:.4f}")
    
    # 打印每折结果
    for r in all_fold_results:
        if args.task == 'classification':
            print(f"第 {r['fold']} 折 - Loss: {r['val_loss']:.4f}, Acc: {r['val_accuracy']:.4f}, F1: {r['val_f1']:.4f}")
        else:
            print(f"第 {r['fold']} 折 - Loss: {r['val_loss']:.4f}, MAE: {r['val_mae']:.4f}, RMSE: {r['val_rmse']:.4f}")
    
    return all_fold_results, all_histories, best_model_states, device, input_dim, output_dim


def train_final_model(model, features, labels, num_epochs, lr, device, task_type, 
                       batch_size=8, weight_decay=1e-4, scheduler_patience=20, print_freq=10):
    """
    在完整数据集上训练最终模型，用于部署（与 prv_informer.py 一致）
    
    Args:
        model: 模型实例
        features: 特征数据
        labels: 标签数据
        num_epochs: 训练轮数
        lr: 学习率
        device: 设备
        task_type: 任务类型
        batch_size: 批次大小
        weight_decay: 权重衰减
        scheduler_patience: 学习率调度器耐心值
        print_freq: 打印频率
    
    Returns:
        model: 训练完成的模型
        history: 训练历史
    """
    print("开始训练最终模型...")
    
    model.to(device)
    
    # 损失函数
    if task_type == 'classification':
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()
    
    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=scheduler_patience
    )
    
    # 创建数据加载器
    if task_type == 'classification':
        dataset = TensorDataset(
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.long)
        )
    else:
        dataset = TensorDataset(
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.float32).reshape(-1, 1)
        )
    
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 记录训练过程
    history = {
        'train_losses': [],
        'train_maes': [],
        'train_rmses': [],
        'train_accs': [],
    }
    
    # 用于保存最佳模型
    best_train_loss = float('inf')
    best_model_state = None
    
    for epoch in range(num_epochs):
        # 训练阶段
        train_result = train_epoch(model, data_loader, criterion, optimizer, device, task_type)
        
        # 保存最佳模型
        if train_result['loss'] < best_train_loss:
            best_train_loss = train_result['loss']
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        # 更新学习率
        scheduler.step(train_result['loss'])
        
        # 保存历史
        history['train_losses'].append(train_result['loss'])
        history['train_maes'].append(train_result.get('mae', 0))
        history['train_accs'].append(train_result.get('accuracy', 0))
        
        # 打印进度
        if (epoch + 1) % print_freq == 0:
            current_lr = optimizer.param_groups[0]['lr']
            if task_type == 'classification':
                print(f'Epoch: {epoch+1}/{num_epochs}, '
                      f'Train Loss: {train_result["loss"]:.4f}, '
                      f'Acc: {train_result.get("accuracy", 0):.4f}, '
                      f'LR: {current_lr:.6f}')
            else:
                print(f'Epoch: {epoch+1}/{num_epochs}, '
                      f'Train Loss: {train_result["loss"]:.4f}, '
                      f'MAE: {train_result.get("mae", 0):.4f}, '
                      f'LR: {current_lr:.6f}')
    
    # 加载最佳模型权重
    model.load_state_dict(best_model_state)
    
    print(f"最终模型训练完成! 最佳训练损失: {best_train_loss:.4f}")
    
    return model, history


def evaluate_model(model, test_loader, device, task_type):
    """
    评估模型（与 prv_informer.py 一致）
    
    Args:
        model: 模型实例
        test_loader: 测试数据加载器
        device: 设备
        task_type: 任务类型
    
    Returns:
        result: 评估结果字典
        predictions: 预测值列表
        targets: 目标值列表
    """
    if task_type == 'classification':
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()
    
    val_result = validate_epoch(model, test_loader, criterion, device, task_type)
    
    predictions = val_result['predictions']
    targets = val_result['targets']
    
    print(f'Test Loss: {val_result["loss"]:.4f}', end='')
    if task_type == 'classification':
        print(f', Accuracy: {val_result.get("accuracy", 0):.4f}', end='')
        print(f', F1: {val_result.get("f1", 0):.4f}')
    else:
        print(f', MAE: {val_result.get("mae", 0):.4f}', end='')
        print(f', RMSE: {val_result.get("rmse", 0):.4f}')
    
    return val_result, predictions, targets


def load_and_evaluate_model(model_path, test_loader, device, task_type):
    """
    加载已保存的模型并对测试集进行评估（与 prv_informer.py 一致）
    
    Args:
        model_path: 模型文件路径
        test_loader: 测试数据加载器
        device: 设备类型
        task_type: 任务类型
    
    Returns:
        result: 评估结果
        predictions: 预测值
        targets: 目标值
    """
    print(f"正在从 {model_path} 加载模型...")
    
    # 添加对自定义模型类的安全声明
    torch.serialization.add_safe_globals([
        LSTM, GRU, BiLSTM, TemporalConvNet, TimeSeriesTransformer, Informer
    ])
    
    # 直接加载完整模型
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.to(device)
    
    print("模型加载完成，开始评估...")
    
    # 评估模型
    result, predictions, targets = evaluate_model(model, test_loader, device, task_type)
    
    print("模型评估完成!")
    return result, predictions, targets


def main():
    """主函数（与 prv_informer.py 流程一致）"""
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    print("=" * 60)
    print("基准模型测试训练 - 与 prv_informer.py 流程一致")
    print("=" * 60)
    print(f"模型: {args.model.upper()}")
    print(f"数据类型: {args.data_type.upper()}")
    print(f"任务类型: {args.task}")
    print(f"隐藏维度: {args.hidden_dim}")
    print(f"层数: {args.num_layers}")
    print(f"训练轮数: {args.epochs}")
    print(f"批次大小: {args.batch_size}")
    print(f"学习率: {args.lr}")
    print(f"K折数: {args.k_folds}")
    
    # 解析选择的情绪
    selected_emotions = None
    if args.emotions:
        selected_emotions = [e.strip() for e in args.emotions.split(',')]
        print(f"选择情绪: {selected_emotions}")
    
    # 加载数据
    print("\n===== 加载数据 =====")
    try:
        features, stress_labels, emotion_labels, scaler = load_data(
            args.data_dir, args.data_type, selected_emotions
        )
    except FileNotFoundError as e:
        print(f"错误: {e}")
        return
    
    print(f"数据形状: {features.shape}")
    print(f"压力标签形状: {stress_labels.shape}")
    print(f"情绪标签分布: {dict(zip(*np.unique(emotion_labels, return_counts=True)))}")
    
    # K折交叉验证训练
    print("\n===== 开始 K折交叉验证训练 =====")
    fold_results, all_histories, best_model_states, device, input_dim, output_dim = train_kfold(
        args, features, stress_labels, emotion_labels
    )
    
    # 绘制所有折的平均训练过程曲线
    print("\n===== 绘制训练过程曲线 =====")
    avg_train_losses = np.mean([h['train_losses'] for h in all_histories], axis=0)
    avg_val_losses = np.mean([h['val_losses'] for h in all_histories], axis=0)
    avg_train_maes = np.mean([h['train_maes'] for h in all_histories], axis=0)
    avg_val_maes = np.mean([h['val_maes'] for h in all_histories], axis=0)
    avg_train_rmses = np.mean([h.get('train_rmses', [0]*len(h['train_losses'])) for h in all_histories], axis=0)
    avg_val_rmses = np.mean([h['val_rmses'] for h in all_histories], axis=0)
    
    model_name = f"{args.data_type}_{args.model}_{args.task}"
    plot_training_process(
        avg_train_losses.tolist(), avg_val_losses.tolist(),
        avg_train_maes.tolist(), avg_val_maes.tolist(),
        avg_train_rmses.tolist(), avg_val_rmses.tolist(),
        model_name, save_dir="results", show=False
    )
    print(f"训练过程曲线已保存到 results/{model_name}_training_process.png")
    
    # 绘制各折结果对比图
    plot_fold_comparison(fold_results, model_name, save_dir="results", show=False)
    print(f"各折结果对比图已保存到 results/{model_name}_fold_comparison.png")
    
    # 选择验证集表现最好的模型
    print("\n===== 选择最佳模型并评估测试集 =====")
    best_fold_index = np.argmin([result['val_loss'] for result in fold_results])
    print(f"选择第 {best_fold_index+1} 折的模型作为最佳模型")
    
    # 创建模型并加载最佳权重
    best_model = create_model(
        args.model, input_dim, output_dim,
        args.hidden_dim, args.num_layers, args.task
    )
    best_model.load_state_dict(best_model_states[best_fold_index])
    best_model.to(device)
    
    # 创建测试数据加载器（使用部分数据作为测试集）
    # 这里我们使用 10% 的数据作为测试集
    from sklearn.model_selection import train_test_split
    _, X_test, _, y_stress_test, _, y_emotion_test = train_test_split(
        features, stress_labels, emotion_labels,
        test_size=0.1, random_state=args.seed, stratify=emotion_labels
    )
    
    if args.task == 'classification':
        test_tensors = [
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_emotion_test, dtype=torch.long)
        ]
    else:
        test_tensors = [
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_stress_test, dtype=torch.float32).reshape(-1, 1)
        ]
    
    test_dataset = TensorDataset(*test_tensors)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # 评估模型
    test_result, predictions, targets = evaluate_model(
        best_model, test_loader, device, args.task
    )
    
    # 绘制预测结果（仅回归任务）
    if args.task == 'regression':
        plot_predictions(predictions, targets, model_name, save_dir="results", show=False)
        print(f"预测结果图已保存到 results/{model_name}_predictions.png")
    
    print("\n模型训练和评估完成!")
    if args.task == 'classification':
        print(f"测试集 - Loss: {test_result['loss']:.4f}, "
              f"Accuracy: {test_result.get('accuracy', 0):.4f}, "
              f"F1: {test_result.get('f1', 0):.4f}")
    else:
        print(f"测试集 - Loss: {test_result['loss']:.4f}, "
              f"MAE: {test_result.get('mae', 0):.4f}, "
              f"RMSE: {test_result.get('rmse', 0):.4f}")
    
    # 训练用于部署的最终模型
    print("\n===== 开始训练用于部署的最终模型 =====")
    
    # 合并训练集和验证集用于最终训练（不包含测试集）
    X_full = features  # 使用全部数据进行最终训练
    if args.task == 'classification':
        y_full = emotion_labels
    else:
        y_full = stress_labels
    
    print(f"完整数据集大小: {X_full.shape}")
    
    # 创建新的模型用于最终训练
    final_model = create_model(
        args.model, input_dim, output_dim,
        args.hidden_dim, args.num_layers, args.task
    )
    
    # 使用与交叉验证相同的超参数训练最终模型
    final_model, final_history = train_final_model(
        final_model, X_full, y_full,
        args.epochs, args.lr, device, args.task,
        batch_size=args.batch_size
    )
    
    # 保存最终模型
    model_save_path = f"{args.data_type}_{args.model}_{args.task}_model.pth"
    torch.save(final_model, model_save_path)
    print(f"最终模型已保存到: {model_save_path}")
    
    # 示例：加载保存的模型并重新评估测试集
    print("\n===== 示例：加载保存的模型并重新评估测试集 =====")
    loaded_result, loaded_predictions, loaded_targets = load_and_evaluate_model(
        model_save_path, test_loader, device, args.task
    )
    
    print("加载模型评估结果:")
    if args.task == 'classification':
        print(f"测试集 - Loss: {loaded_result['loss']:.4f}, "
              f"Accuracy: {loaded_result.get('accuracy', 0):.4f}, "
              f"F1: {loaded_result.get('f1', 0):.4f}")
    else:
        print(f"测试集 - Loss: {loaded_result['loss']:.4f}, "
              f"MAE: {loaded_result.get('mae', 0):.4f}, "
              f"RMSE: {loaded_result.get('rmse', 0):.4f}")
    
    print("\n===== 训练完成 =====")


if __name__ == "__main__":
    main()
