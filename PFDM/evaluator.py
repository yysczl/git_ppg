"""PPG-Former-DualStream 评估器
包含模型评估、测试和结果分析功能
"""

import os
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

from utils import Logger, calculate_metrics, calculate_classification_metrics


class Evaluator:
    """
    模型评估器
    
    支持：
    1. 回归任务评估（压力预测）- MAE, RMSE
    2. 分类任务评估（情绪分类）- Accuracy, Precision, Recall, F1-Score
    3. 多任务评估
    4. 结果可视化
    """
    
    def __init__(self, model: nn.Module, device: str = 'cpu', logger: Logger = None,
                 task_type: str = 'regression', train_mode: str = 'dual_stream'):
        """
        Args:
            model: 模型实例
            device: 设备
            logger: 日志记录器
            task_type: 任务类型 (regression/classification/multi_task)
            train_mode: 训练模式
        """
        self.model = model
        self.device = torch.device(device)
        self.logger = logger
        self.task_type = task_type
        self.train_mode = train_mode
        
        self.model.to(self.device)
        self.model.eval()
        
        # 损失函数
        self.mse_criterion = nn.MSELoss()
        self.ce_criterion = nn.CrossEntropyLoss()
    
    def _log(self, message: str):
        """记录日志"""
        if self.logger:
            self.logger.info(message)
        else:
            print(message)
    
    def evaluate_regression(self, test_loader: DataLoader,
                            return_predictions: bool = True
                            ) -> Dict[str, Any]:
        """
        评估回归任务（压力预测）
        
        评估指标：MAE, RMSE, R², 相关系数
        
        Args:
            test_loader: 测试数据加载器
            return_predictions: 是否返回预测值
        
        Returns:
            包含评估指标的字典
        """
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        total_loss = 0
        batch_count = 0
        
        with torch.no_grad():
            for batch_data in test_loader:
                # 解析数据
                batch_info = self._parse_batch_data(batch_data)
                input_data = batch_info['input_data']
                stress_target = batch_info['stress_target']
                prv_data = batch_info.get('prv_data', None)
                
                if stress_target is None:
                    continue
                
                # 前向传播
                output = self._forward(input_data, prv_data)
                if isinstance(output, tuple):
                    output = output[0]  # 取压力预测
                
                # 计算损失
                loss = self.mse_criterion(output.squeeze(), stress_target.squeeze())
                total_loss += loss.item()
                
                # 收集预测和目标
                pred = output.squeeze().cpu().numpy()
                target = stress_target.squeeze().cpu().numpy()
                
                if np.ndim(pred) == 0:
                    all_predictions.append(pred.item())
                    all_targets.append(target.item())
                else:
                    all_predictions.extend(pred.tolist())
                    all_targets.extend(target.tolist())
                
                batch_count += 1
        
        if batch_count == 0:
            return {}
        
        # 转换为numpy数组
        predictions = np.array(all_predictions)
        targets = np.array(all_targets)
        
        # 计算评估指标
        avg_loss = total_loss / batch_count
        metrics = calculate_metrics(predictions, targets)
        metrics['loss'] = avg_loss
        
        self._log(f"\n===== 压力回归评估结果 =====")
        self._log(f"  Loss: {avg_loss:.4f}")
        self._log(f"  MAE: {metrics['mae']:.4f}")
        self._log(f"  RMSE: {metrics['rmse']:.4f}")
        self._log(f"  R²: {metrics['r2']:.4f}")
        self._log(f"  相关系数: {metrics['correlation']:.4f}")
        
        if return_predictions:
            metrics['predictions'] = predictions.tolist()
            metrics['targets'] = targets.tolist()
        
        return metrics
    
    def _parse_batch_data(self, batch_data: tuple) -> dict:
        """解析批次数据"""
        result = {
            'input_data': None,
            'prv_data': None,
            'stress_target': None,
            'emotion_target': None
        }
        
        if len(batch_data) == 4:
            # PPG + PRV + stress + emotion 或 input + stress + emotion + extra
            if self.train_mode in ['dual_stream', 'multi_task']:
                ppg_data, prv_data, stress_target, emotion_target = batch_data
                result['input_data'] = ppg_data.to(self.device)
                result['prv_data'] = prv_data.to(self.device)
                result['stress_target'] = stress_target.to(self.device)
                result['emotion_target'] = emotion_target.to(self.device)
            else:
                input_data, stress_target, emotion_target, _ = batch_data
                result['input_data'] = input_data.to(self.device)
                result['stress_target'] = stress_target.to(self.device)
                result['emotion_target'] = emotion_target.to(self.device)
        elif len(batch_data) == 3:
            if self.train_mode in ['dual_stream', 'multi_task']:
                ppg_data, prv_data, stress_target = batch_data
                result['input_data'] = ppg_data.to(self.device)
                result['prv_data'] = prv_data.to(self.device)
                result['stress_target'] = stress_target.to(self.device)
            else:
                input_data, stress_target, emotion_target = batch_data
                result['input_data'] = input_data.to(self.device)
                result['stress_target'] = stress_target.to(self.device)
                if isinstance(emotion_target, torch.Tensor):
                    result['emotion_target'] = emotion_target.to(self.device)
        elif len(batch_data) == 2:
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
    
    def _forward(self, input_data, prv_data):
        """前向传播"""
        if prv_data is not None and hasattr(self.model, 'compute_loss'):
            return self.model(input_data, prv_data)
        else:
            return self.model(input_data)
    
    def evaluate_classification(self, test_loader: DataLoader,
                                emotion_labels: List[str] = None
                                ) -> Dict[str, Any]:
        """
        评估分类任务（情绪分类）
        
        评估指标：Accuracy, Precision, Recall, F1-Score
        
        Args:
            test_loader: 测试数据加载器
            emotion_labels: 情绪类别名称
        
        Returns:
            包含评估指标的字典
        """
        if emotion_labels is None:
            emotion_labels = ['Anxiety', 'Happy', 'Peace', 'Sad', 'Stress']
        
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        total_loss = 0
        batch_count = 0
        
        with torch.no_grad():
            for batch_data in test_loader:
                # 解析数据
                batch_info = self._parse_batch_data(batch_data)
                input_data = batch_info['input_data']
                emotion_target = batch_info['emotion_target']
                prv_data = batch_info.get('prv_data', None)
                
                if emotion_target is None:
                    continue
                
                # 前向传播
                output = self._forward(input_data, prv_data)
                if isinstance(output, tuple):
                    emotion_pred = output[1]  # 取情绪预测
                else:
                    emotion_pred = output  # 单任务分类
                
                # 计算损失
                loss = self.ce_criterion(emotion_pred, emotion_target)
                total_loss += loss.item()
                
                # 收集预测和目标
                pred = emotion_pred.argmax(dim=1).cpu().numpy()
                target = emotion_target.cpu().numpy()
                
                all_predictions.extend(pred.tolist())
                all_targets.extend(target.tolist())
                
                batch_count += 1
        
        if batch_count == 0:
            self._log("警告: 没有情绪分类数据")
            return {}
        
        predictions = np.array(all_predictions)
        targets = np.array(all_targets)
        
        # 计算评估指标
        avg_loss = total_loss / batch_count
        accuracy = accuracy_score(targets, predictions)
        
        # Precision, Recall, F1
        precision_macro = precision_score(targets, predictions, average='macro', zero_division=0)
        precision_weighted = precision_score(targets, predictions, average='weighted', zero_division=0)
        recall_macro = recall_score(targets, predictions, average='macro', zero_division=0)
        recall_weighted = recall_score(targets, predictions, average='weighted', zero_division=0)
        f1_macro = f1_score(targets, predictions, average='macro', zero_division=0)
        f1_weighted = f1_score(targets, predictions, average='weighted', zero_division=0)
        
        # 混淆矩阵
        cm = confusion_matrix(targets, predictions)
        
        # 分类报告
        report = classification_report(
            targets, predictions,
            target_names=emotion_labels,
            output_dict=True,
            zero_division=0
        )
        
        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'precision_weighted': precision_weighted,
            'recall_macro': recall_macro,
            'recall_weighted': recall_weighted,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'predictions': predictions.tolist(),
            'targets': targets.tolist()
        }
        
        self._log(f"\n===== 情绪分类评估结果 =====")
        self._log(f"  Loss: {avg_loss:.4f}")
        self._log(f"  Accuracy: {accuracy:.4f}")
        self._log(f"  Precision (macro): {precision_macro:.4f}")
        self._log(f"  Recall (macro): {recall_macro:.4f}")
        self._log(f"  F1-Score (macro): {f1_macro:.4f}")
        self._log(f"  F1-Score (weighted): {f1_weighted:.4f}")
        
        # 打印每个类别的指标
        self._log(f"\n  各类别详细指标:")
        for i, label in enumerate(emotion_labels):
            if label in report:
                self._log(f"    {label}: Precision={report[label]['precision']:.4f}, "
                         f"Recall={report[label]['recall']:.4f}, "
                         f"F1={report[label]['f1-score']:.4f}")
        
        return metrics
    
    def evaluate_multi_task(self, test_loader: DataLoader,
                            emotion_labels: List[str] = None
                            ) -> Dict[str, Any]:
        """
        多任务评估（回归 + 分类）
        
        Args:
            test_loader: 测试数据加载器
            emotion_labels: 情绪类别名称
        
        Returns:
            包含所有评估指标的字典
        """
        self._log("\n===== 多任务评估 =====")
        
        # 回归评估
        regression_metrics = self.evaluate_regression(test_loader, return_predictions=True)
        
        # 检查是否有分类数据
        has_emotion = False
        for batch_data in test_loader:
            if len(batch_data) >= 4:
                has_emotion = True
                break
        
        if has_emotion:
            classification_metrics = self.evaluate_classification(test_loader, emotion_labels)
        else:
            classification_metrics = {}
        
        return {
            'regression': regression_metrics,
            'classification': classification_metrics
        }


def evaluate_model(model: nn.Module, test_loader: DataLoader,
                   criterion: nn.Module = None, device: str = 'cpu',
                   return_predictions: bool = True) -> Tuple[float, float, float, List, List]:
    """
    简单的模型评估函数（向后兼容）
    
    Args:
        model: 模型实例
        test_loader: 测试数据加载器
        criterion: 损失函数
        device: 设备
        return_predictions: 是否返回预测值
    
    Returns:
        test_loss, test_mae, test_rmse, predictions, targets
    """
    if criterion is None:
        criterion = nn.MSELoss()
    
    device = torch.device(device)
    model.to(device)
    model.eval()
    
    all_predictions = []
    all_targets = []
    total_loss = 0
    total_mae = 0
    total_rmse = 0
    batch_count = 0
    
    with torch.no_grad():
        for batch_data in test_loader:
            # 解析数据
            if len(batch_data) >= 3:
                ppg_data = batch_data[0].to(device)
                prv_data = batch_data[1].to(device)
                stress_target = batch_data[2].to(device)
            else:
                ppg_data = batch_data[0].to(device)
                stress_target = batch_data[1].to(device)
                prv_data = None
            
            # 确保输入维度正确
            if ppg_data.dim() == 2:
                ppg_data = ppg_data.unsqueeze(-1)
            if prv_data is not None and prv_data.dim() == 2:
                prv_data = prv_data.unsqueeze(-1)
            
            # 前向传播
            if hasattr(model, 'compute_loss') and prv_data is not None:
                output, _ = model(ppg_data, prv_data)
            else:
                output = model(ppg_data)
            
            # 计算损失
            loss = criterion(output.squeeze(), stress_target.squeeze())
            total_loss += loss.item()
            
            # 计算指标
            diff = torch.abs(output.squeeze() - stress_target.squeeze())
            mae = torch.mean(diff)
            rmse = torch.sqrt(torch.mean(diff ** 2))
            
            total_mae += mae.item()
            total_rmse += rmse.item()
            
            # 收集预测和目标
            pred = output.squeeze().cpu().numpy()
            target = stress_target.squeeze().cpu().numpy()
            
            if np.ndim(pred) == 0:
                all_predictions.append(pred.item())
                all_targets.append(target.item())
            else:
                all_predictions.extend(pred.tolist())
                all_targets.extend(target.tolist())
            
            batch_count += 1
    
    avg_loss = total_loss / batch_count
    avg_mae = total_mae / batch_count
    avg_rmse = total_rmse / batch_count
    
    print(f"测试结果 - Loss: {avg_loss:.4f}, MAE: {avg_mae:.4f}, RMSE: {avg_rmse:.4f}")
    
    return avg_loss, avg_mae, avg_rmse, all_predictions, all_targets


def compare_models(models: Dict[str, nn.Module], test_loader: DataLoader,
                   device: str = 'cpu') -> Dict[str, Dict]:
    """
    对比多个模型的性能
    
    Args:
        models: 模型字典 {模型名称: 模型实例}
        test_loader: 测试数据加载器
        device: 设备
    
    Returns:
        各模型的评估结果
    """
    results = {}
    
    print("\n===== 模型对比 =====")
    
    for name, model in models.items():
        print(f"\n评估模型: {name}")
        
        evaluator = Evaluator(model, device)
        metrics = evaluator.evaluate_regression(test_loader, return_predictions=False)
        
        results[name] = metrics
    
    # 打印对比表格
    print("\n===== 对比结果 =====")
    print(f"{'模型名称':<25} {'Loss':>10} {'MAE':>10} {'RMSE':>10} {'R²':>10}")
    print("-" * 65)
    
    for name, metrics in results.items():
        print(f"{name:<25} {metrics['loss']:>10.4f} {metrics['mae']:>10.4f} "
              f"{metrics['rmse']:>10.4f} {metrics['r2']:>10.4f}")
    
    return results


def ablation_study(base_model_class, base_params: dict,
                   ablation_configs: Dict[str, dict],
                   test_loader: DataLoader, device: str = 'cpu') -> Dict[str, Dict]:
    """
    消融实验
    
    Args:
        base_model_class: 基础模型类
        base_params: 基础模型参数
        ablation_configs: 消融配置 {配置名称: 修改的参数}
        test_loader: 测试数据加载器
        device: 设备
    
    Returns:
        各配置的评估结果
    """
    results = {}
    
    print("\n===== 消融实验 =====")
    
    # 评估基础模型
    print("\n评估基础模型...")
    base_model = base_model_class(**base_params)
    evaluator = Evaluator(base_model, device)
    base_metrics = evaluator.evaluate_regression(test_loader, return_predictions=False)
    results['baseline'] = base_metrics
    
    # 评估各消融配置
    for config_name, config_changes in ablation_configs.items():
        print(f"\n评估消融配置: {config_name}")
        
        # 合并参数
        ablation_params = base_params.copy()
        ablation_params.update(config_changes)
        
        # 创建和评估模型
        ablation_model = base_model_class(**ablation_params)
        evaluator = Evaluator(ablation_model, device)
        metrics = evaluator.evaluate_regression(test_loader, return_predictions=False)
        
        results[config_name] = metrics
    
    # 打印消融结果
    print("\n===== 消融结果 =====")
    print(f"{'配置名称':<30} {'Loss':>10} {'MAE':>10} {'RMSE':>10} {'相对变化':>12}")
    print("-" * 72)
    
    baseline_rmse = results['baseline']['rmse']
    
    for name, metrics in results.items():
        relative_change = (metrics['rmse'] - baseline_rmse) / baseline_rmse * 100
        change_str = f"{relative_change:+.2f}%"
        
        print(f"{name:<30} {metrics['loss']:>10.4f} {metrics['mae']:>10.4f} "
              f"{metrics['rmse']:>10.4f} {change_str:>12}")
    
    return results


class ResultAnalyzer:
    """
    结果分析器
    """
    
    def __init__(self, predictions: np.ndarray, targets: np.ndarray,
                 emotion_preds: np.ndarray = None, emotion_targets: np.ndarray = None):
        """
        Args:
            predictions: 压力预测值
            targets: 压力真实值
            emotion_preds: 情绪预测值（可选）
            emotion_targets: 情绪真实值（可选）
        """
        self.predictions = np.array(predictions)
        self.targets = np.array(targets)
        self.emotion_preds = np.array(emotion_preds) if emotion_preds is not None else None
        self.emotion_targets = np.array(emotion_targets) if emotion_targets is not None else None
    
    def get_regression_summary(self) -> Dict[str, float]:
        """获取回归结果摘要"""
        errors = self.predictions - self.targets
        abs_errors = np.abs(errors)
        
        summary = {
            'mae': np.mean(abs_errors),
            'rmse': np.sqrt(np.mean(errors ** 2)),
            'mse': np.mean(errors ** 2),
            'r2': r2_score(self.targets, self.predictions),
            'correlation': np.corrcoef(self.predictions, self.targets)[0, 1],
            'max_error': np.max(abs_errors),
            'min_error': np.min(abs_errors),
            'std_error': np.std(errors),
            'median_error': np.median(abs_errors)
        }
        
        return summary
    
    def get_error_distribution(self, bins: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """获取误差分布"""
        errors = self.predictions - self.targets
        hist, bin_edges = np.histogram(errors, bins=bins)
        return hist, bin_edges
    
    def get_classification_summary(self, emotion_labels: List[str] = None) -> Dict:
        """获取分类结果摘要"""
        if self.emotion_preds is None or self.emotion_targets is None:
            return {}
        
        if emotion_labels is None:
            emotion_labels = ['Anxiety', 'Happy', 'Peace', 'Sad', 'Stress']
        
        summary = {
            'accuracy': accuracy_score(self.emotion_targets, self.emotion_preds),
            'f1_macro': f1_score(self.emotion_targets, self.emotion_preds, average='macro'),
            'f1_weighted': f1_score(self.emotion_targets, self.emotion_preds, average='weighted'),
            'confusion_matrix': confusion_matrix(self.emotion_targets, self.emotion_preds).tolist(),
            'per_class_accuracy': {}
        }
        
        # 计算每个类别的准确率
        for i, label in enumerate(emotion_labels):
            mask = self.emotion_targets == i
            if np.sum(mask) > 0:
                class_acc = np.mean(self.emotion_preds[mask] == i)
                summary['per_class_accuracy'][label] = class_acc
        
        return summary
    
    def print_summary(self):
        """打印结果摘要"""
        print("\n===== 回归任务结果摘要 =====")
        reg_summary = self.get_regression_summary()
        
        for key, value in reg_summary.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
        
        if self.emotion_preds is not None:
            print("\n===== 分类任务结果摘要 =====")
            cls_summary = self.get_classification_summary()
            
            print(f"  Accuracy: {cls_summary['accuracy']:.4f}")
            print(f"  F1 (macro): {cls_summary['f1_macro']:.4f}")
            print(f"  F1 (weighted): {cls_summary['f1_weighted']:.4f}")
            
            if cls_summary['per_class_accuracy']:
                print("\n  各类别准确率:")
                for label, acc in cls_summary['per_class_accuracy'].items():
                    print(f"    {label}: {acc:.4f}")
