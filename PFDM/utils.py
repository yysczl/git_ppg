"""
PPG-Former-DualStream 工具函数
包含数据加载、绘图、日志等功能
"""

import os
import json
import random
import logging
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# ============ 随机种子设置 ============
def set_seed(seed: int = 42):
    """设置随机种子以确保结果可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============ 日志系统 ============
class Logger:
    """训练日志记录器"""
    
    def __init__(self, log_dir: str, experiment_name: str, log_level: str = "INFO"):
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 创建日志目录
        os.makedirs(log_dir, exist_ok=True)
        
        # 设置日志文件路径
        self.log_file = os.path.join(log_dir, f"{experiment_name}_{self.timestamp}.log")
        
        # 配置logger
        self.logger = logging.getLogger(experiment_name)
        self.logger.setLevel(getattr(logging, log_level))
        
        # 清除已有的handler
        self.logger.handlers.clear()
        
        # 文件handler
        file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        file_handler.setLevel(getattr(logging, log_level))
        
        # 控制台handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level))
        
        # 格式化
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # 训练历史记录
        self.history = {
            'train_losses': [],
            'val_losses': [],
            'train_maes': [],
            'val_maes': [],
            'train_rmses': [],
            'val_rmses': [],
            'fold_results': [],
            'config': None
        }
    
    def info(self, message: str):
        """记录信息"""
        self.logger.info(message)
    
    def warning(self, message: str):
        """记录警告"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """记录错误"""
        self.logger.error(message)
    
    def log_config(self, config: Any):
        """记录实验配置"""
        self.history['config'] = str(config)
        self.info(f"实验配置: {config}")
    
    def log_epoch(self, epoch: int, train_loss: float, val_loss: float,
                  train_mae: float, val_mae: float, train_rmse: float, val_rmse: float,
                  lr: float = None):
        """记录每个epoch的结果"""
        self.history['train_losses'].append(train_loss)
        self.history['val_losses'].append(val_loss)
        self.history['train_maes'].append(train_mae)
        self.history['val_maes'].append(val_mae)
        self.history['train_rmses'].append(train_rmse)
        self.history['val_rmses'].append(val_rmse)
        
        msg = f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
        msg += f"Train MAE={train_mae:.4f}, Val MAE={val_mae:.4f}, "
        msg += f"Train RMSE={train_rmse:.4f}, Val RMSE={val_rmse:.4f}"
        if lr is not None:
            msg += f", LR={lr:.6f}"
        self.info(msg)
    
    def log_fold_result(self, fold: int, val_loss: float, val_mae: float, val_rmse: float):
        """记录每折的结果"""
        result = {
            'fold': fold,
            'val_loss': val_loss,
            'val_mae': val_mae,
            'val_rmse': val_rmse
        }
        self.history['fold_results'].append(result)
        self.info(f"第 {fold} 折结果 - Val Loss: {val_loss:.4f}, MAE: {val_mae:.4f}, RMSE: {val_rmse:.4f}")
    
    def log_test_result(self, test_loss: float, test_mae: float, test_rmse: float,
                        emotion_acc: float = None):
        """记录测试结果"""
        self.history['test_loss'] = test_loss
        self.history['test_mae'] = test_mae
        self.history['test_rmse'] = test_rmse
        
        msg = f"测试结果 - Loss: {test_loss:.4f}, MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}"
        if emotion_acc is not None:
            self.history['emotion_acc'] = emotion_acc
            msg += f", Emotion Acc: {emotion_acc:.4f}"
        self.info(msg)
    
    def save_history(self, save_path: str = None):
        """保存训练历史"""
        if save_path is None:
            save_path = os.path.join(self.log_dir, f"{self.experiment_name}_{self.timestamp}_history.json")
        
        # 转换numpy数组为列表
        history_to_save = {}
        for key, value in self.history.items():
            if isinstance(value, np.ndarray):
                history_to_save[key] = value.tolist()
            elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], np.ndarray):
                history_to_save[key] = [v.tolist() if isinstance(v, np.ndarray) else v for v in value]
            else:
                history_to_save[key] = value
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(history_to_save, f, indent=4, ensure_ascii=False)
        
        self.info(f"训练历史已保存到: {save_path}")
        return save_path


# ============ 数据集类 ============
class PPGDataset(Dataset):
    """PPG数据集"""
    
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32).reshape(-1, 1)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class MultiModalDataset(Dataset):
    """多模态数据集（PPG + PRV）"""
    
    def __init__(self, ppg_features: np.ndarray, prv_features: np.ndarray,
                 stress_labels: np.ndarray, emotion_labels: np.ndarray = None):
        self.ppg_features = torch.tensor(ppg_features, dtype=torch.float32)
        self.prv_features = torch.tensor(prv_features, dtype=torch.float32)
        self.stress_labels = torch.tensor(stress_labels, dtype=torch.float32).reshape(-1, 1)
        
        if emotion_labels is not None:
            self.emotion_labels = torch.tensor(emotion_labels, dtype=torch.long)
        else:
            self.emotion_labels = None
    
    def __len__(self):
        return len(self.ppg_features)
    
    def __getitem__(self, idx):
        if self.emotion_labels is not None:
            return (self.ppg_features[idx], self.prv_features[idx],
                    self.stress_labels[idx], self.emotion_labels[idx])
        return self.ppg_features[idx], self.prv_features[idx], self.stress_labels[idx]


# ============ 数据加载函数 ============
def load_ppg_data(file_path: str, test_size: float = 0.1, val_size: float = 0.1
                  ) -> Tuple[np.ndarray, ...]:
    """
    加载PPG数据
    
    Args:
        file_path: 数据文件路径
        test_size: 测试集比例
        val_size: 验证集比例
    
    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test, scaler
    """
    try:
        data = pd.read_csv(file_path, header=None, skiprows=1)
    except Exception as e:
        print(f"读取CSV文件时出错: {e}")
        data = pd.read_csv(file_path, header=None, encoding='latin1')
    
    print(f"PPG数据形状: {data.shape}")
    
    # 前1800列为特征，最后一列为标签
    features = data.iloc[:, :1800].values
    labels = data.iloc[:, -1].values
    
    print(f"PPG特征形状: {features.shape}, 标签形状: {labels.shape}")
    
    # 标准化
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # 划分数据集
    X_train, X_temp, y_train, y_temp = train_test_split(
        features_scaled, labels, test_size=test_size + val_size, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=test_size / (test_size + val_size), random_state=42
    )
    
    print(f"训练集: {X_train.shape}, 验证集: {X_val.shape}, 测试集: {X_test.shape}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, scaler


def load_prv_data(file_path: str, test_size: float = 0.1, val_size: float = 0.1
                  ) -> Tuple[np.ndarray, ...]:
    """
    加载PRV数据
    
    Args:
        file_path: 数据文件路径
        test_size: 测试集比例
        val_size: 验证集比例
    
    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test, scaler
    """
    try:
        data = pd.read_csv(file_path)
    except Exception as e:
        print(f"读取PRV CSV文件时出错: {e}")
        data = pd.read_csv(file_path, encoding='latin1')
    
    print(f"PRV数据形状: {data.shape}")
    
    # 前80列为特征，最后一列为标签
    features = data.iloc[:, :80].values
    labels = data.iloc[:, -1].values
    
    print(f"PRV特征形状: {features.shape}, 标签形状: {labels.shape}")
    
    # 标准化
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # 划分数据集
    X_train, X_temp, y_train, y_temp = train_test_split(
        features_scaled, labels, test_size=test_size + val_size, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=test_size / (test_size + val_size), random_state=42
    )
    
    print(f"PRV训练集: {X_train.shape}, 验证集: {X_val.shape}, 测试集: {X_test.shape}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, scaler


def create_data_loaders(X_ppg: np.ndarray, X_prv: Optional[np.ndarray],
                        y_stress: np.ndarray, y_emotion: Optional[np.ndarray] = None,
                        batch_size: int = 8, shuffle: bool = True) -> DataLoader:
    """创建数据加载器"""
    if X_prv is not None:
        if y_emotion is not None:
            dataset = TensorDataset(
                torch.tensor(X_ppg, dtype=torch.float32),
                torch.tensor(X_prv, dtype=torch.float32),
                torch.tensor(y_stress, dtype=torch.float32).reshape(-1, 1),
                torch.tensor(y_emotion, dtype=torch.long)
            )
        else:
            dataset = TensorDataset(
                torch.tensor(X_ppg, dtype=torch.float32),
                torch.tensor(X_prv, dtype=torch.float32),
                torch.tensor(y_stress, dtype=torch.float32).reshape(-1, 1)
            )
    else:
        dataset = TensorDataset(
            torch.tensor(X_ppg, dtype=torch.float32),
            torch.tensor(y_stress, dtype=torch.float32).reshape(-1, 1)
        )
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# ============ 绘图函数 ============
def plot_training_process(train_losses: List[float], val_losses: List[float],
                          train_maes: List[float], val_maes: List[float],
                          train_rmses: List[float], val_rmses: List[float],
                          model_type: str, save_dir: str = "results",
                          show: bool = True):
    """绘制训练过程曲线"""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 绘制损失曲线
    axes[0].plot(train_losses, label='Train Loss', color='blue')
    axes[0].plot(val_losses, label='Val Loss', color='red')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss vs. Epoch')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 绘制MAE曲线
    axes[1].plot(train_maes, label='Train MAE', color='blue')
    axes[1].plot(val_maes, label='Val MAE', color='red')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE')
    axes[1].set_title('MAE vs. Epoch')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 绘制RMSE曲线
    axes[2].plot(train_rmses, label='Train RMSE', color='blue')
    axes[2].plot(val_rmses, label='Val RMSE', color='red')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('RMSE')
    axes[2].set_title('RMSE vs. Epoch')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{model_type}_training_process.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    plt.close()
    
    return save_path


def plot_predictions(predictions: List[float], targets: List[float],
                     model_type: str, save_dir: str = "results",
                     show: bool = True):
    """绘制预测结果散点图"""
    os.makedirs(save_dir, exist_ok=True)
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.scatter(targets, predictions, alpha=0.5, c='blue', edgecolors='none')
    
    # 绘制对角线
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal')
    
    ax.set_xlabel('True Values')
    ax.set_ylabel('Predictions')
    ax.set_title(f'{model_type} - Predictions vs. True Values')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 添加统计信息
    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    correlation = np.corrcoef(predictions, targets)[0, 1]
    
    text = f'MAE: {mae:.4f}\nRMSE: {rmse:.4f}\nCorr: {correlation:.4f}'
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{model_type}_predictions.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    plt.close()
    
    return save_path


def plot_fold_comparison(fold_results: List[Dict], model_type: str,
                         save_dir: str = "results", show: bool = True):
    """绘制各折结果对比图"""
    os.makedirs(save_dir, exist_ok=True)
    
    folds = [r['fold'] for r in fold_results]
    val_losses = [r['val_loss'] for r in fold_results]
    val_maes = [r['val_mae'] for r in fold_results]
    val_rmses = [r['val_rmse'] for r in fold_results]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(folds))
    width = 0.25
    
    bars1 = ax.bar(x - width, val_losses, width, label='Val Loss', color='blue', alpha=0.7)
    bars2 = ax.bar(x, val_maes, width, label='Val MAE', color='green', alpha=0.7)
    bars3 = ax.bar(x + width, val_rmses, width, label='Val RMSE', color='red', alpha=0.7)
    
    ax.set_xlabel('Fold')
    ax.set_ylabel('Value')
    ax.set_title(f'{model_type} - K-Fold Cross Validation Results')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Fold {f}' for f in folds])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加平均线
    ax.axhline(y=np.mean(val_losses), color='blue', linestyle='--', alpha=0.5)
    ax.axhline(y=np.mean(val_maes), color='green', linestyle='--', alpha=0.5)
    ax.axhline(y=np.mean(val_rmses), color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{model_type}_fold_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    plt.close()
    
    return save_path


# ============ 模型工具函数 ============
def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    """统计模型参数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def save_model(model: torch.nn.Module, save_path: str, config: Any = None):
    """保存模型"""
    save_dict = {
        'model_state_dict': model.state_dict(),
        'config': str(config) if config else None
    }
    torch.save(save_dict, save_path)
    print(f"模型已保存到: {save_path}")


def load_model(model: torch.nn.Module, load_path: str, device: str = 'cpu'):
    """加载模型"""
    checkpoint = torch.load(load_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"模型已从 {load_path} 加载")
    return model


# ============ 评估指标计算 ============
def calculate_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """计算评估指标"""
    predictions = np.array(predictions).flatten()
    targets = np.array(targets).flatten()
    
    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    mse = np.mean((predictions - targets) ** 2)
    
    # 计算R²
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    # 计算相关系数
    correlation = np.corrcoef(predictions, targets)[0, 1] if len(predictions) > 1 else 0
    
    return {
        'mae': mae,
        'rmse': rmse,
        'mse': mse,
        'r2': r2,
        'correlation': correlation
    }
