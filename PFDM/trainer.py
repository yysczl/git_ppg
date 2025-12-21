"""PPG-Former-DualStream 训练器
包含K折交叉验证训练逻辑、单任务和多任务训练支持
"""

import os
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold, StratifiedKFold

from utils import Logger, set_seed, calculate_classification_metrics


class Trainer:
    """
    模型训练器
    
    支持：
    1. 单折训练
    2. K折交叉验证
    3. 单任务学习（回归/分类）
    4. 多任务学习
    5. 早停机制
    6. 学习率调度
    7. 日志记录
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
        
        # 获取任务类型
        self.task_type = getattr(config.training, 'task_type', 'regression')
        self.train_mode = getattr(config.training, 'train_mode', 'dual_stream')
        
        self.model.to(self.device)
        
        # 损失函数
        if self.task_type == 'classification':
            self.criterion = nn.CrossEntropyLoss()
        else:  # regression or multi_task
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
        self.early_stop_counter = 0
        
        # 训练历史
        self.history = {
            'train_losses': [],
            'val_losses': [],
            'train_maes': [],
            'val_maes': [],
            'train_rmses': [],
            'val_rmses': [],
            # 分类任务指标
            'train_accs': [],
            'val_accs': [],
            'train_f1s': [],
            'val_f1s': [],
        }
    
    def _log(self, message: str):
        """记录日志"""
        if self.logger:
            self.logger.info(message)
        else:
            print(message)
    
    def train_epoch(self, train_loader: DataLoader, use_emotion: bool = False
                    ) -> Dict[str, float]:
        """
        训练一个epoch
        
        Args:
            train_loader: 训练数据加载器
            use_emotion: 是否使用情绪分类任务
        
        Returns:
            包含各种指标的字典
        """
        self.model.train()
        total_loss = 0
        total_mae = 0
        total_rmse = 0
        total_correct = 0
        total_samples = 0
        all_preds = []
        all_targets = []
        batch_count = 0
        
        for batch_data in train_loader:
            # 解析批次数据
            batch_info = self._parse_batch_data(batch_data, use_emotion)
            input_data = batch_info['input_data']
            stress_target = batch_info['stress_target']
            emotion_target = batch_info['emotion_target']
            prv_data = batch_info.get('prv_data', None)
            
            self.optimizer.zero_grad()
            
            # 前向传播和损失计算
            loss, outputs = self._forward_and_loss(
                input_data, prv_data, stress_target, emotion_target, use_emotion
            )
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.config.training.max_grad_norm
            )
            
            self.optimizer.step()
            
            # 计算指标
            total_loss += loss.item()
            metrics = self._compute_batch_metrics(
                outputs, stress_target, emotion_target, use_emotion
            )
            total_mae += metrics.get('mae', 0)
            total_rmse += metrics.get('rmse', 0)
            total_correct += metrics.get('correct', 0)
            total_samples += metrics.get('samples', 0)
            
            if emotion_target is not None and self.task_type in ['classification', 'multi_task']:
                all_preds.extend(metrics.get('preds', []))
                all_targets.extend(emotion_target.cpu().numpy().tolist())
            
            batch_count += 1
        
        result = {
            'loss': total_loss / batch_count,
            'mae': total_mae / batch_count if self.task_type != 'classification' else 0,
            'rmse': total_rmse / batch_count if self.task_type != 'classification' else 0,
        }
        
        if total_samples > 0:
            result['accuracy'] = total_correct / total_samples
            if len(all_preds) > 0:
                cls_metrics = calculate_classification_metrics(
                    np.array(all_preds), np.array(all_targets)
                )
                result['f1'] = cls_metrics['f1_macro']
        
        return result
    
    def _parse_batch_data(self, batch_data: tuple, use_emotion: bool = False) -> dict:
        """解析批次数据"""
        result = {
            'input_data': None,
            'prv_data': None,
            'stress_target': None,
            'emotion_target': None
        }
        
        if use_emotion and len(batch_data) == 4:
            # PPG + PRV + stress + emotion
            ppg_data, prv_data, stress_target, emotion_target = batch_data
            result['input_data'] = ppg_data.to(self.device)
            result['prv_data'] = prv_data.to(self.device)
            result['stress_target'] = stress_target.to(self.device)
            result['emotion_target'] = emotion_target.to(self.device)
        elif len(batch_data) == 4:
            # PPG/PRV + stress + emotion (单模态 + 多任务)
            input_data, stress_target, emotion_target, _ = batch_data
            result['input_data'] = input_data.to(self.device)
            result['stress_target'] = stress_target.to(self.device)
            result['emotion_target'] = emotion_target.to(self.device)
        elif len(batch_data) == 3:
            if self.train_mode in ['dual_stream', 'multi_task']:
                # PPG + PRV + stress
                ppg_data, prv_data, stress_target = batch_data
                result['input_data'] = ppg_data.to(self.device)
                result['prv_data'] = prv_data.to(self.device)
                result['stress_target'] = stress_target.to(self.device)
            else:
                # 单模态 + stress + emotion
                input_data, stress_target, emotion_target = batch_data
                result['input_data'] = input_data.to(self.device)
                result['stress_target'] = stress_target.to(self.device)
                if isinstance(emotion_target, torch.Tensor):
                    result['emotion_target'] = emotion_target.to(self.device)
        elif len(batch_data) == 2:
            # 单模态 + stress/emotion
            input_data, target = batch_data
            result['input_data'] = input_data.to(self.device)
            if self.task_type == 'classification':
                result['emotion_target'] = target.to(self.device)
            else:
                result['stress_target'] = target.to(self.device)
        
        # 确保输入维度正确
        if result['input_data'] is not None and result['input_data'].dim() == 2:
            result['input_data'] = result['input_data'].unsqueeze(-1)
        if result['prv_data'] is not None and result['prv_data'].dim() == 2:
            result['prv_data'] = result['prv_data'].unsqueeze(-1)
        
        return result
    
    def _forward_and_loss(self, input_data, prv_data, stress_target, 
                          emotion_target, use_emotion: bool) -> Tuple[torch.Tensor, dict]:
        """前向传播并计算损失"""
        outputs = {}
        
        if self.task_type == 'classification':
            # 单任务分类
            if hasattr(self.model, 'compute_loss'):
                loss, _, _ = self.model.compute_loss(
                    input_data, stress_target, emotion_target
                )
                output = self.model(input_data)
            else:
                output = self.model(input_data)
                loss = self.criterion(output, emotion_target)
            outputs['emotion_pred'] = output
        elif self.task_type == 'multi_task' or (use_emotion and self.train_mode == 'multi_task'):
            # 多任务学习
            if prv_data is not None:
                loss, _, _ = self.model.compute_loss(
                    input_data, prv_data, stress_target, emotion_target
                )
                stress_pred, emotion_pred = self.model(input_data, prv_data)
            else:
                loss, _, _ = self.model.compute_loss(
                    input_data, stress_target, emotion_target
                )
                result = self.model(input_data)
                if isinstance(result, tuple):
                    stress_pred, emotion_pred = result
                else:
                    stress_pred = result
                    emotion_pred = None
            outputs['stress_pred'] = stress_pred
            outputs['emotion_pred'] = emotion_pred
        else:
            # 单任务回归
            if prv_data is not None and hasattr(self.model, 'compute_loss'):
                loss, _, _ = self.model.compute_loss(
                    input_data, prv_data, stress_target, emotion_target
                )
                output, _ = self.model(input_data, prv_data)
            elif hasattr(self.model, 'compute_loss'):
                loss, _, _ = self.model.compute_loss(
                    input_data, stress_target, emotion_target
                )
                output = self.model(input_data)
            else:
                output = self.model(input_data)
                loss = self.criterion(output.squeeze(), stress_target.squeeze())
            outputs['stress_pred'] = output
        
        return loss, outputs
    
    def _compute_batch_metrics(self, outputs: dict, stress_target, emotion_target,
                               use_emotion: bool) -> dict:
        """计算批次指标"""
        metrics = {}
        
        with torch.no_grad():
            if 'stress_pred' in outputs and stress_target is not None:
                pred = outputs['stress_pred'].squeeze()
                target = stress_target.squeeze()
                diff = torch.abs(pred - target)
                metrics['mae'] = torch.mean(diff).item()
                metrics['rmse'] = torch.sqrt(torch.mean(diff ** 2)).item()
            
            if 'emotion_pred' in outputs and emotion_target is not None:
                pred = outputs['emotion_pred']
                if pred is not None:
                    pred_labels = torch.argmax(pred, dim=1)
                    correct = (pred_labels == emotion_target).sum().item()
                    metrics['correct'] = correct
                    metrics['samples'] = len(emotion_target)
                    metrics['preds'] = pred_labels.cpu().numpy().tolist()
        
        return metrics
    
    def validate_epoch(self, val_loader: DataLoader, use_emotion: bool = False
                       ) -> Dict[str, float]:
        """
        验证一个epoch
        
        Args:
            val_loader: 验证数据加载器
            use_emotion: 是否使用情绪分类任务
        
        Returns:
            包含各种指标的字典
        """
        self.model.eval()
        total_loss = 0
        total_mae = 0
        total_rmse = 0
        total_correct = 0
        total_samples = 0
        all_preds = []
        all_targets = []
        batch_count = 0
        
        with torch.no_grad():
            for batch_data in val_loader:
                # 解析批次数据
                batch_info = self._parse_batch_data(batch_data, use_emotion)
                input_data = batch_info['input_data']
                stress_target = batch_info['stress_target']
                emotion_target = batch_info['emotion_target']
                prv_data = batch_info.get('prv_data', None)
                
                # 前向传播和损失计算
                loss, outputs = self._forward_and_loss(
                    input_data, prv_data, stress_target, emotion_target, use_emotion
                )
                
                total_loss += loss.item()
                
                # 计算指标
                metrics = self._compute_batch_metrics(
                    outputs, stress_target, emotion_target, use_emotion
                )
                total_mae += metrics.get('mae', 0)
                total_rmse += metrics.get('rmse', 0)
                total_correct += metrics.get('correct', 0)
                total_samples += metrics.get('samples', 0)
                
                if emotion_target is not None and self.task_type in ['classification', 'multi_task']:
                    all_preds.extend(metrics.get('preds', []))
                    all_targets.extend(emotion_target.cpu().numpy().tolist())
                
                batch_count += 1
        
        result = {
            'loss': total_loss / batch_count,
            'mae': total_mae / batch_count if self.task_type != 'classification' else 0,
            'rmse': total_rmse / batch_count if self.task_type != 'classification' else 0,
        }
        
        if total_samples > 0:
            result['accuracy'] = total_correct / total_samples
            if len(all_preds) > 0:
                cls_metrics = calculate_classification_metrics(
                    np.array(all_preds), np.array(all_targets)
                )
                result['f1'] = cls_metrics['f1_macro']
        
        return result
    
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
        early_stopping = getattr(self.config.training, 'early_stopping', False)
        early_stopping_patience = getattr(self.config.training, 'early_stopping_patience', 50)
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch + 1
            
            # 训练
            train_metrics = self.train_epoch(train_loader, use_emotion)
            
            # 验证
            val_metrics = self.validate_epoch(val_loader, use_emotion)
            
            # 更新学习率
            self.scheduler.step(val_metrics['loss'])
            
            # 保存最佳模型
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.best_model_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1
            
            # 记录历史
            self.history['train_losses'].append(train_metrics['loss'])
            self.history['val_losses'].append(val_metrics['loss'])
            self.history['train_maes'].append(train_metrics.get('mae', 0))
            self.history['val_maes'].append(val_metrics.get('mae', 0))
            self.history['train_rmses'].append(train_metrics.get('rmse', 0))
            self.history['val_rmses'].append(val_metrics.get('rmse', 0))
            self.history['train_accs'].append(train_metrics.get('accuracy', 0))
            self.history['val_accs'].append(val_metrics.get('accuracy', 0))
            self.history['train_f1s'].append(train_metrics.get('f1', 0))
            self.history['val_f1s'].append(val_metrics.get('f1', 0))
            
            # 日志记录
            if self.logger:
                current_lr = self.optimizer.param_groups[0]['lr']
                self._log_epoch_metrics(epoch + 1, train_metrics, val_metrics, current_lr)
            elif (epoch + 1) % print_freq == 0:
                self._print_epoch_metrics(epoch + 1, num_epochs, train_metrics, val_metrics)
            
            # 早停检查
            if early_stopping and self.early_stop_counter >= early_stopping_patience:
                if self.logger:
                    self.logger.info(f"早停触发！连续 {early_stopping_patience} 个epoch验证损失未改善")
                else:
                    print(f"早停触发！连续 {early_stopping_patience} 个epoch验证损失未改善")
                break
        
        return self.history
    
    def _log_epoch_metrics(self, epoch: int, train_metrics: dict, val_metrics: dict, lr: float):
        """记录epoch指标到日志"""
        if self.task_type == 'classification':
            msg = f"Epoch {epoch}: Train Loss={train_metrics['loss']:.4f}, "
            msg += f"Val Loss={val_metrics['loss']:.4f}, "
            msg += f"Train Acc={train_metrics.get('accuracy', 0):.4f}, "
            msg += f"Val Acc={val_metrics.get('accuracy', 0):.4f}, "
            msg += f"Train F1={train_metrics.get('f1', 0):.4f}, "
            msg += f"Val F1={val_metrics.get('f1', 0):.4f}, LR={lr:.6f}"
        else:
            self.logger.log_epoch(
                epoch, train_metrics['loss'], val_metrics['loss'],
                train_metrics.get('mae', 0), val_metrics.get('mae', 0),
                train_metrics.get('rmse', 0), val_metrics.get('rmse', 0), lr
            )
            return
        self.logger.info(msg)
    
    def _print_epoch_metrics(self, epoch: int, total_epochs: int, 
                             train_metrics: dict, val_metrics: dict):
        """打印epoch指标"""
        current_lr = self.optimizer.param_groups[0]['lr']
        print(f'Epoch: {epoch}/{total_epochs}')
        
        if self.task_type == 'classification':
            print(f'  Train Loss: {train_metrics["loss"]:.4f}, Acc: {train_metrics.get("accuracy", 0):.4f}')
            print(f'  Val Loss: {val_metrics["loss"]:.4f}, Acc: {val_metrics.get("accuracy", 0):.4f}')
        else:
            print(f'  Train Loss: {train_metrics["loss"]:.4f}, MAE: {train_metrics.get("mae", 0):.4f}, RMSE: {train_metrics.get("rmse", 0):.4f}')
            print(f'  Val Loss: {val_metrics["loss"]:.4f}, MAE: {val_metrics.get("mae", 0):.4f}, RMSE: {val_metrics.get("rmse", 0):.4f}')
        print(f'  Learning Rate: {current_lr:.6f}')
    
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
                      X_train: np.ndarray, X_prv_train: Optional[np.ndarray],
                      y_stress_train: np.ndarray, y_emotion_train: Optional[np.ndarray],
                      X_val: np.ndarray, X_prv_val: Optional[np.ndarray],
                      y_stress_val: np.ndarray, y_emotion_val: Optional[np.ndarray],
                      config, logger: Logger = None,
                      use_emotion: bool = False) -> Tuple[dict, dict, dict]:
    """
    训练单折模型
    
    Returns:
        history, best_model_state, fold_result
    """
    device = torch.device(config.device if hasattr(config, 'device') else 'cpu')
    task_type = getattr(config.training, 'task_type', 'regression')
    train_mode = getattr(config.training, 'train_mode', 'dual_stream')
    
    # 创建数据加载器
    train_tensors = [torch.tensor(X_train, dtype=torch.float32)]
    val_tensors = [torch.tensor(X_val, dtype=torch.float32)]
    
    if X_prv_train is not None and train_mode in ['dual_stream', 'multi_task']:
        train_tensors.append(torch.tensor(X_prv_train, dtype=torch.float32))
        val_tensors.append(torch.tensor(X_prv_val, dtype=torch.float32))
    
    if task_type == 'classification':
        # 分类任务：只需要情绪标签
        train_tensors.append(torch.tensor(y_emotion_train, dtype=torch.long))
        val_tensors.append(torch.tensor(y_emotion_val, dtype=torch.long))
    elif task_type == 'multi_task' or use_emotion:
        # 多任务：同时需要压力和情绪标签
        train_tensors.append(torch.tensor(y_stress_train, dtype=torch.float32).reshape(-1, 1))
        train_tensors.append(torch.tensor(y_emotion_train, dtype=torch.long))
        val_tensors.append(torch.tensor(y_stress_val, dtype=torch.float32).reshape(-1, 1))
        val_tensors.append(torch.tensor(y_emotion_val, dtype=torch.long))
    else:
        # 回归任务：只需要压力标签
        train_tensors.append(torch.tensor(y_stress_train, dtype=torch.float32).reshape(-1, 1))
        val_tensors.append(torch.tensor(y_stress_val, dtype=torch.float32).reshape(-1, 1))
    
    train_dataset = TensorDataset(*train_tensors)
    val_dataset = TensorDataset(*val_tensors)
    
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
        use_emotion=use_emotion or task_type == 'multi_task'
    )
    
    # 获取最终验证结果
    trainer.load_best_model()
    val_metrics = trainer.validate_epoch(val_loader, use_emotion or task_type == 'multi_task')
    
    fold_result = {
        'val_loss': val_metrics['loss'],
        'val_mae': val_metrics.get('mae', 0),
        'val_rmse': val_metrics.get('rmse', 0),
        'val_accuracy': val_metrics.get('accuracy', 0),
        'val_f1': val_metrics.get('f1', 0),
    }
    
    return history, trainer.best_model_state, fold_result


def train_kfold(model_class, model_params: dict,
                X_data: np.ndarray, X_prv: Optional[np.ndarray],
                y_stress: np.ndarray, y_emotion: Optional[np.ndarray],
                config, logger: Logger = None,
                use_emotion: bool = False,
                stratify_by_emotion: bool = True) -> Tuple[List[dict], List[dict], List[dict]]:
    """
    K折交叉验证训练
    
    Args:
        model_class: 模型类
        model_params: 模型参数
        X_data: 主输入数据 (PPG 或 PRV)
        X_prv: PRV数据（可选）
        y_stress: 压力标签
        y_emotion: 情绪标签（可选）
        config: 配置对象
        logger: 日志记录器
        use_emotion: 是否使用情绪任务
        stratify_by_emotion: 是否按情绪分层抽样
    
    Returns:
        all_histories, best_model_states, fold_results
    """
    k_folds = config.training.k_folds
    task_type = getattr(config.training, 'task_type', 'regression')
    
    # 如果是分类任务，使用分层K折
    if stratify_by_emotion and y_emotion is not None and task_type == 'classification':
        kfold = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=config.training.seed)
        split_data = y_emotion
    else:
        kfold = KFold(n_splits=k_folds, shuffle=True, random_state=config.training.seed)
        split_data = X_data
    
    all_histories = []
    best_model_states = []
    fold_results = []
    
    if logger:
        logger.info(f"开始 {k_folds} 折交叉验证训练")
    else:
        print(f"\n开始 {k_folds} 折交叉验证训练")
    
    for fold, (train_ids, val_ids) in enumerate(kfold.split(X_data, split_data if isinstance(split_data, np.ndarray) and len(split_data.shape) == 1 else None)):
        fold_num = fold + 1
        
        if logger:
            logger.info(f"\n===== 第 {fold_num}/{k_folds} 折 =====")
            logger.info(f"训练集大小: {len(train_ids)}, 验证集大小: {len(val_ids)}")
        else:
            print(f"\n===== 第 {fold_num}/{k_folds} 折 =====")
            print(f"训练集大小: {len(train_ids)}, 验证集大小: {len(val_ids)}")
        
        # 划分数据
        X_train, X_val = X_data[train_ids], X_data[val_ids]
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
            X_train, X_prv_train, y_stress_train, y_emotion_train,
            X_val, X_prv_val, y_stress_val, y_emotion_val,
            config, logger, use_emotion
        )
        
        result['fold'] = fold_num
        
        all_histories.append(history)
        best_model_states.append(best_state)
        fold_results.append(result)
        
        # 记录折结果
        if logger:
            if task_type == 'classification':
                logger.info(f"第 {fold_num} 折结果 - Val Loss: {result['val_loss']:.4f}, "
                           f"Acc: {result.get('val_accuracy', 0):.4f}, F1: {result.get('val_f1', 0):.4f}")
            else:
                logger.log_fold_result(fold_num, result['val_loss'], result['val_mae'], result['val_rmse'])
        else:
            if task_type == 'classification':
                print(f"第 {fold_num} 折结果 - Val Loss: {result['val_loss']:.4f}, "
                      f"Acc: {result.get('val_accuracy', 0):.4f}, F1: {result.get('val_f1', 0):.4f}")
            else:
                print(f"第 {fold_num} 折结果 - Val Loss: {result['val_loss']:.4f}, "
                      f"MAE: {result['val_mae']:.4f}, RMSE: {result['val_rmse']:.4f}")
    
    # 计算平均结果
    avg_val_loss = np.mean([r['val_loss'] for r in fold_results])
    
    if logger:
        logger.info(f"\n===== K折交叉验证结果 =====")
        logger.info(f"平均验证损失: {avg_val_loss:.4f}")
        if task_type == 'classification':
            avg_acc = np.mean([r.get('val_accuracy', 0) for r in fold_results])
            avg_f1 = np.mean([r.get('val_f1', 0) for r in fold_results])
            logger.info(f"平均验证Accuracy: {avg_acc:.4f}")
            logger.info(f"平均验证F1: {avg_f1:.4f}")
        else:
            avg_val_mae = np.mean([r['val_mae'] for r in fold_results])
            avg_val_rmse = np.mean([r['val_rmse'] for r in fold_results])
            logger.info(f"平均验证MAE: {avg_val_mae:.4f}")
            logger.info(f"平均验证RMSE: {avg_val_rmse:.4f}")
    else:
        print(f"\n===== K折交叉验证结果 =====")
        print(f"平均验证损失: {avg_val_loss:.4f}")
        if task_type == 'classification':
            avg_acc = np.mean([r.get('val_accuracy', 0) for r in fold_results])
            avg_f1 = np.mean([r.get('val_f1', 0) for r in fold_results])
            print(f"平均验证Accuracy: {avg_acc:.4f}")
            print(f"平均验证F1: {avg_f1:.4f}")
        else:
            avg_val_mae = np.mean([r['val_mae'] for r in fold_results])
            avg_val_rmse = np.mean([r['val_rmse'] for r in fold_results])
            print(f"平均验证MAE: {avg_val_mae:.4f}")
            print(f"平均验证RMSE: {avg_val_rmse:.4f}")
    
    return all_histories, best_model_states, fold_results


def get_average_history(all_histories: List[dict]) -> dict:
    """
    计算所有折的平均训练历史
    
    处理不同折可能有不同长度的情况（由于早停）
    """
    if not all_histories:
        return {}
    
    avg_history = {}
    
    keys = ['train_losses', 'val_losses', 'train_maes', 'val_maes', 
            'train_rmses', 'val_rmses', 'train_accs', 'val_accs', 
            'train_f1s', 'val_f1s']
    
    for key in keys:
        # 获取所有折的该指标
        values = [h.get(key, []) for h in all_histories]
        
        # 过滤掉空列表
        values = [v for v in values if len(v) > 0]
        
        if not values:
            avg_history[key] = []
            continue
        
        # 找到最短的历史长度（处理早停导致的不同长度）
        min_len = min(len(v) for v in values)
        
        # 截取到最短长度后计算平均
        truncated_values = [v[:min_len] for v in values]
        
        try:
            avg_history[key] = np.mean(truncated_values, axis=0).tolist()
        except Exception:
            # 如果还是失败，取第一折的历史
            avg_history[key] = values[0] if values else []
    
    return avg_history


def get_best_fold(fold_results: List[dict]) -> int:
    """获取最佳折的索引"""
    val_losses = [r['val_loss'] for r in fold_results]
    return int(np.argmin(val_losses))
