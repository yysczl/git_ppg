"""
PPG-Former-DualStream 训练器
包含K折交叉验证训练逻辑和多任务训练支持
"""

import os
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold

from utils import Logger, set_seed


class Trainer:
    """
    模型训练器
    
    支持：
    1. 单折训练
    2. K折交叉验证
    3. 多任务学习
    4. 早停机制
    5. 学习率调度
    6. 日志记录
    """
    
    def __init__(self, model: nn.Module, config: Any, logger: Logger = None,
                 device: str = 'cpu'):
        """
        Args:
            model: 模型实例
            config: 配置对象
            logger: 日志记录器
            device: 设备
        """
        self.model = model
        self.config = config
        self.logger = logger
        self.device = torch.device(device)
        
        self.model.to(self.device)
        
        # 损失函数
        self.criterion = nn.MSELoss()
        
        # 优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=config.training.scheduler_factor,
            patience=config.training.scheduler_patience
        )
        
        # 训练状态
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.current_epoch = 0
        
        # 训练历史
        self.history = {
            'train_losses': [],
            'val_losses': [],
            'train_maes': [],
            'val_maes': [],
            'train_rmses': [],
            'val_rmses': []
        }
    
    def _log(self, message: str):
        """记录日志"""
        if self.logger:
            self.logger.info(message)
        else:
            print(message)
    
    def train_epoch(self, train_loader: DataLoader, use_emotion: bool = False
                    ) -> Tuple[float, float, float]:
        """
        训练一个epoch
        
        Args:
            train_loader: 训练数据加载器
            use_emotion: 是否使用情绪分类任务
        
        Returns:
            train_loss, train_mae, train_rmse
        """
        self.model.train()
        total_loss = 0
        total_mae = 0
        total_rmse = 0
        batch_count = 0
        
        for batch_data in train_loader:
            # 解析批次数据
            if use_emotion and len(batch_data) == 4:
                ppg_data, prv_data, stress_target, emotion_target = batch_data
                ppg_data = ppg_data.to(self.device)
                prv_data = prv_data.to(self.device)
                stress_target = stress_target.to(self.device)
                emotion_target = emotion_target.to(self.device)
            elif len(batch_data) == 3:
                ppg_data, prv_data, stress_target = batch_data
                ppg_data = ppg_data.to(self.device)
                prv_data = prv_data.to(self.device)
                stress_target = stress_target.to(self.device)
                emotion_target = None
            else:
                ppg_data, stress_target = batch_data[0], batch_data[1]
                ppg_data = ppg_data.to(self.device)
                stress_target = stress_target.to(self.device)
                prv_data = None
                emotion_target = None
            
            # 确保输入维度正确
            if ppg_data.dim() == 2:
                ppg_data = ppg_data.unsqueeze(-1)
            if prv_data is not None and prv_data.dim() == 2:
                prv_data = prv_data.unsqueeze(-1)
            
            self.optimizer.zero_grad()
            
            # 前向传播
            if hasattr(self.model, 'compute_loss'):
                total_batch_loss, stress_loss, emotion_loss = self.model.compute_loss(
                    ppg_data, prv_data, stress_target, emotion_target
                )
                output, _ = self.model(ppg_data, prv_data)
            else:
                output = self.model(ppg_data)
                total_batch_loss = self.criterion(output.squeeze(), stress_target.squeeze())
                stress_loss = total_batch_loss
            
            # 反向传播
            total_batch_loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.config.training.max_grad_norm
            )
            
            self.optimizer.step()
            
            # 计算指标
            total_loss += stress_loss.item()
            
            with torch.no_grad():
                diff = torch.abs(output.squeeze() - stress_target.squeeze())
                mae = torch.mean(diff)
                rmse = torch.sqrt(torch.mean(diff ** 2))
                
                total_mae += mae.item()
                total_rmse += rmse.item()
            
            batch_count += 1
        
        avg_loss = total_loss / batch_count
        avg_mae = total_mae / batch_count
        avg_rmse = total_rmse / batch_count
        
        return avg_loss, avg_mae, avg_rmse
    
    def validate_epoch(self, val_loader: DataLoader, use_emotion: bool = False
                       ) -> Tuple[float, float, float]:
        """
        验证一个epoch
        
        Args:
            val_loader: 验证数据加载器
            use_emotion: 是否使用情绪分类任务
        
        Returns:
            val_loss, val_mae, val_rmse
        """
        self.model.eval()
        total_loss = 0
        total_mae = 0
        total_rmse = 0
        batch_count = 0
        
        with torch.no_grad():
            for batch_data in val_loader:
                # 解析批次数据
                if use_emotion and len(batch_data) == 4:
                    ppg_data, prv_data, stress_target, emotion_target = batch_data
                    ppg_data = ppg_data.to(self.device)
                    prv_data = prv_data.to(self.device)
                    stress_target = stress_target.to(self.device)
                elif len(batch_data) == 3:
                    ppg_data, prv_data, stress_target = batch_data
                    ppg_data = ppg_data.to(self.device)
                    prv_data = prv_data.to(self.device)
                    stress_target = stress_target.to(self.device)
                else:
                    ppg_data, stress_target = batch_data[0], batch_data[1]
                    ppg_data = ppg_data.to(self.device)
                    stress_target = stress_target.to(self.device)
                    prv_data = None
                
                # 确保输入维度正确
                if ppg_data.dim() == 2:
                    ppg_data = ppg_data.unsqueeze(-1)
                if prv_data is not None and prv_data.dim() == 2:
                    prv_data = prv_data.unsqueeze(-1)
                
                # 前向传播
                if hasattr(self.model, 'compute_loss'):
                    output, _ = self.model(ppg_data, prv_data)
                else:
                    output = self.model(ppg_data)
                
                loss = self.criterion(output.squeeze(), stress_target.squeeze())
                total_loss += loss.item()
                
                # 计算指标
                diff = torch.abs(output.squeeze() - stress_target.squeeze())
                mae = torch.mean(diff)
                rmse = torch.sqrt(torch.mean(diff ** 2))
                
                total_mae += mae.item()
                total_rmse += rmse.item()
                
                batch_count += 1
        
        avg_loss = total_loss / batch_count
        avg_mae = total_mae / batch_count
        avg_rmse = total_rmse / batch_count
        
        return avg_loss, avg_mae, avg_rmse
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              num_epochs: int = None, use_emotion: bool = False
              ) -> Dict[str, List[float]]:
        """
        训练模型
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            num_epochs: 训练轮数
            use_emotion: 是否使用情绪分类任务
        
        Returns:
            训练历史字典
        """
        if num_epochs is None:
            num_epochs = self.config.training.num_epochs
        
        print_freq = self.config.log.print_freq if hasattr(self.config, 'log') else 10
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch + 1
            
            # 训练
            train_loss, train_mae, train_rmse = self.train_epoch(
                train_loader, use_emotion
            )
            
            # 验证
            val_loss, val_mae, val_rmse = self.validate_epoch(
                val_loader, use_emotion
            )
            
            # 更新学习率
            self.scheduler.step(val_loss)
            
            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
            
            # 记录历史
            self.history['train_losses'].append(train_loss)
            self.history['val_losses'].append(val_loss)
            self.history['train_maes'].append(train_mae)
            self.history['val_maes'].append(val_mae)
            self.history['train_rmses'].append(train_rmse)
            self.history['val_rmses'].append(val_rmse)
            
            # 日志记录
            if self.logger:
                current_lr = self.optimizer.param_groups[0]['lr']
                self.logger.log_epoch(
                    epoch + 1, train_loss, val_loss,
                    train_mae, val_mae, train_rmse, val_rmse, current_lr
                )
            elif (epoch + 1) % print_freq == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f'Epoch: {epoch + 1}/{num_epochs}')
                print(f'  Train Loss: {train_loss:.4f}, MAE: {train_mae:.4f}, RMSE: {train_rmse:.4f}')
                print(f'  Val Loss: {val_loss:.4f}, MAE: {val_mae:.4f}, RMSE: {val_rmse:.4f}')
                print(f'  Learning Rate: {current_lr:.6f}')
        
        return self.history
    
    def load_best_model(self):
        """加载最佳模型"""
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            self._log("已加载最佳模型")
    
    def save_model(self, save_path: str):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history
        }, save_path)
        self._log(f"模型已保存到: {save_path}")


def train_single_fold(model_class, model_params: dict,
                      X_ppg_train: np.ndarray, X_prv_train: Optional[np.ndarray],
                      y_stress_train: np.ndarray, y_emotion_train: Optional[np.ndarray],
                      X_ppg_val: np.ndarray, X_prv_val: Optional[np.ndarray],
                      y_stress_val: np.ndarray, y_emotion_val: Optional[np.ndarray],
                      config, logger: Logger = None,
                      use_emotion: bool = False) -> Tuple[dict, dict, dict]:
    """
    训练单折模型
    
    Returns:
        history, best_model_state, fold_result
    """
    device = torch.device(config.device if hasattr(config, 'device') else 'cpu')
    
    # 创建数据加载器
    if X_prv_train is not None:
        if use_emotion and y_emotion_train is not None:
            train_dataset = TensorDataset(
                torch.tensor(X_ppg_train, dtype=torch.float32),
                torch.tensor(X_prv_train, dtype=torch.float32),
                torch.tensor(y_stress_train, dtype=torch.float32).reshape(-1, 1),
                torch.tensor(y_emotion_train, dtype=torch.long)
            )
            val_dataset = TensorDataset(
                torch.tensor(X_ppg_val, dtype=torch.float32),
                torch.tensor(X_prv_val, dtype=torch.float32),
                torch.tensor(y_stress_val, dtype=torch.float32).reshape(-1, 1),
                torch.tensor(y_emotion_val, dtype=torch.long)
            )
        else:
            train_dataset = TensorDataset(
                torch.tensor(X_ppg_train, dtype=torch.float32),
                torch.tensor(X_prv_train, dtype=torch.float32),
                torch.tensor(y_stress_train, dtype=torch.float32).reshape(-1, 1)
            )
            val_dataset = TensorDataset(
                torch.tensor(X_ppg_val, dtype=torch.float32),
                torch.tensor(X_prv_val, dtype=torch.float32),
                torch.tensor(y_stress_val, dtype=torch.float32).reshape(-1, 1)
            )
    else:
        train_dataset = TensorDataset(
            torch.tensor(X_ppg_train, dtype=torch.float32),
            torch.tensor(y_stress_train, dtype=torch.float32).reshape(-1, 1)
        )
        val_dataset = TensorDataset(
            torch.tensor(X_ppg_val, dtype=torch.float32),
            torch.tensor(y_stress_val, dtype=torch.float32).reshape(-1, 1)
        )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size
    )
    
    # 创建模型
    model = model_class(**model_params)
    
    # 创建训练器
    trainer = Trainer(model, config, logger, device)
    
    # 训练
    history = trainer.train(
        train_loader, val_loader,
        num_epochs=config.training.num_epochs,
        use_emotion=use_emotion
    )
    
    # 获取最终验证结果
    trainer.load_best_model()
    val_loss, val_mae, val_rmse = trainer.validate_epoch(val_loader, use_emotion)
    
    fold_result = {
        'val_loss': val_loss,
        'val_mae': val_mae,
        'val_rmse': val_rmse
    }
    
    return history, trainer.best_model_state, fold_result


def train_kfold(model_class, model_params: dict,
                X_ppg: np.ndarray, X_prv: Optional[np.ndarray],
                y_stress: np.ndarray, y_emotion: Optional[np.ndarray],
                config, logger: Logger = None,
                use_emotion: bool = False) -> Tuple[List[dict], List[dict], List[dict]]:
    """
    K折交叉验证训练
    
    Args:
        model_class: 模型类
        model_params: 模型参数
        X_ppg: PPG数据
        X_prv: PRV数据（可选）
        y_stress: 压力标签
        y_emotion: 情绪标签（可选）
        config: 配置对象
        logger: 日志记录器
        use_emotion: 是否使用情绪任务
    
    Returns:
        all_histories, best_model_states, fold_results
    """
    k_folds = config.training.k_folds
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=config.training.seed)
    
    all_histories = []
    best_model_states = []
    fold_results = []
    
    if logger:
        logger.info(f"开始 {k_folds} 折交叉验证训练")
    else:
        print(f"\n开始 {k_folds} 折交叉验证训练")
    
    for fold, (train_ids, val_ids) in enumerate(kfold.split(X_ppg)):
        fold_num = fold + 1
        
        if logger:
            logger.info(f"\n===== 第 {fold_num}/{k_folds} 折 =====")
            logger.info(f"训练集大小: {len(train_ids)}, 验证集大小: {len(val_ids)}")
        else:
            print(f"\n===== 第 {fold_num}/{k_folds} 折 =====")
            print(f"训练集大小: {len(train_ids)}, 验证集大小: {len(val_ids)}")
        
        # 划分数据
        X_ppg_train, X_ppg_val = X_ppg[train_ids], X_ppg[val_ids]
        y_stress_train, y_stress_val = y_stress[train_ids], y_stress[val_ids]
        
        if X_prv is not None:
            X_prv_train, X_prv_val = X_prv[train_ids], X_prv[val_ids]
        else:
            X_prv_train, X_prv_val = None, None
        
        if y_emotion is not None:
            y_emotion_train, y_emotion_val = y_emotion[train_ids], y_emotion[val_ids]
        else:
            y_emotion_train, y_emotion_val = None, None
        
        # 训练单折
        history, best_state, result = train_single_fold(
            model_class, model_params,
            X_ppg_train, X_prv_train, y_stress_train, y_emotion_train,
            X_ppg_val, X_prv_val, y_stress_val, y_emotion_val,
            config, logger, use_emotion
        )
        
        result['fold'] = fold_num
        
        all_histories.append(history)
        best_model_states.append(best_state)
        fold_results.append(result)
        
        if logger:
            logger.log_fold_result(fold_num, result['val_loss'], result['val_mae'], result['val_rmse'])
        else:
            print(f"第 {fold_num} 折结果 - Val Loss: {result['val_loss']:.4f}, "
                  f"MAE: {result['val_mae']:.4f}, RMSE: {result['val_rmse']:.4f}")
    
    # 计算平均结果
    avg_val_loss = np.mean([r['val_loss'] for r in fold_results])
    avg_val_mae = np.mean([r['val_mae'] for r in fold_results])
    avg_val_rmse = np.mean([r['val_rmse'] for r in fold_results])
    
    if logger:
        logger.info(f"\n===== K折交叉验证结果 =====")
        logger.info(f"平均验证损失: {avg_val_loss:.4f}")
        logger.info(f"平均验证MAE: {avg_val_mae:.4f}")
        logger.info(f"平均验证RMSE: {avg_val_rmse:.4f}")
    else:
        print(f"\n===== K折交叉验证结果 =====")
        print(f"平均验证损失: {avg_val_loss:.4f}")
        print(f"平均验证MAE: {avg_val_mae:.4f}")
        print(f"平均验证RMSE: {avg_val_rmse:.4f}")
    
    return all_histories, best_model_states, fold_results


def get_average_history(all_histories: List[dict]) -> dict:
    """计算所有折的平均训练历史"""
    avg_history = {}
    
    keys = ['train_losses', 'val_losses', 'train_maes', 'val_maes', 'train_rmses', 'val_rmses']
    
    for key in keys:
        values = [h[key] for h in all_histories]
        avg_history[key] = np.mean(values, axis=0).tolist()
    
    return avg_history


def get_best_fold(fold_results: List[dict]) -> int:
    """获取最佳折的索引"""
    val_losses = [r['val_loss'] for r in fold_results]
    return int(np.argmin(val_losses))
