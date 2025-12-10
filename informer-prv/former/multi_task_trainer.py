"""
多任务学习训练器
支持压力回归 + 情绪分类联合训练
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset

from ppg_former_model import PPGFormer
from dual_stream_model import PPGFormerDualStream, DualStreamOnly


def train_multi_task_model(model, train_loader, val_loader, criterion, optimizer, 
                           num_epochs=50, device='cpu', scheduler=None, use_emotion=False):
    """
    多任务模型训练函数
    """
    train_losses = []
    val_losses = []
    train_maes = []
    val_maes = []
    train_rmses = []
    val_rmses = []
    
    model.to(device)
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_mae = 0
        train_rmse = 0
        batch_count = 0
        
        for batch_data in train_loader:
            if use_emotion and len(batch_data) == 4:
                ppg_data, prv_data, stress_target, emotion_target = batch_data
                ppg_data = ppg_data.to(device)
                prv_data = prv_data.to(device)
                stress_target = stress_target.to(device)
                emotion_target = emotion_target.to(device)
            else:
                ppg_data, stress_target = batch_data[0], batch_data[1]
                ppg_data = ppg_data.to(device)
                stress_target = stress_target.to(device)
                prv_data = None
                emotion_target = None
            
            optimizer.zero_grad()
            
            batch_size, seq_len = ppg_data.shape[:2]
            if ppg_data.dim() == 2:
                ppg_data = ppg_data.unsqueeze(-1)
            if prv_data is not None and prv_data.dim() == 2:
                prv_data = prv_data.unsqueeze(-1)
            
            if hasattr(model, 'compute_loss'):
                total_loss, stress_loss, emotion_loss = model.compute_loss(
                    ppg_data, prv_data, stress_target, emotion_target
                )
                output, _ = model(ppg_data, prv_data)
            else:
                output = model(ppg_data)
                total_loss = criterion(output.squeeze(), stress_target.squeeze())
                stress_loss = total_loss
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += stress_loss.item()
            
            diff = torch.abs(output.squeeze() - stress_target.squeeze())
            mae = torch.mean(diff)
            rmse = torch.sqrt(torch.mean(diff ** 2))
            
            train_mae += mae.item()
            train_rmse += rmse.item()
            batch_count += 1
        
        train_loss /= batch_count
        train_mae /= batch_count
        train_rmse /= batch_count
        
        model.eval()
        val_loss = 0
        val_mae = 0
        val_rmse = 0
        val_batch_count = 0
        
        with torch.no_grad():
            for batch_data in val_loader:
                if use_emotion and len(batch_data) == 4:
                    ppg_data, prv_data, stress_target, emotion_target = batch_data
                    ppg_data = ppg_data.to(device)
                    prv_data = prv_data.to(device)
                    stress_target = stress_target.to(device)
                else:
                    ppg_data, stress_target = batch_data[0], batch_data[1]
                    ppg_data = ppg_data.to(device)
                    stress_target = stress_target.to(device)
                    prv_data = None
                
                if ppg_data.dim() == 2:
                    ppg_data = ppg_data.unsqueeze(-1)
                if prv_data is not None and prv_data.dim() == 2:
                    prv_data = prv_data.unsqueeze(-1)
                
                if hasattr(model, 'compute_loss'):
                    output, _ = model(ppg_data, prv_data)
                else:
                    output = model(ppg_data)
                
                loss = criterion(output.squeeze(), stress_target.squeeze())
                val_loss += loss.item()
                
                diff = torch.abs(output.squeeze() - stress_target.squeeze())
                mae = torch.mean(diff)
                rmse = torch.sqrt(torch.mean(diff ** 2))
                
                val_mae += mae.item()
                val_rmse += rmse.item()
                val_batch_count += 1
        
        val_loss /= val_batch_count
        val_mae /= val_batch_count
        val_rmse /= val_batch_count
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
        
        if scheduler is not None:
            scheduler.step(val_loss)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_maes.append(train_mae)
        val_maes.append(val_mae)
        train_rmses.append(train_rmse)
        val_rmses.append(val_rmse)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch: {epoch+1}/{num_epochs}')
            print(f'Train Loss: {train_loss:.4f}, MAE: {train_mae:.4f}, RMSE: {train_rmse:.4f}')
            print(f'Val Loss: {val_loss:.4f}, MAE: {val_mae:.4f}, RMSE: {val_rmse:.4f}')
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Current Learning Rate: {current_lr:.6f}')
    
    return train_losses, val_losses, train_maes, val_maes, train_rmses, val_rmses, best_model_state


def train_kfold_multi_task(X_ppg, X_prv, y_stress, y_emotion, model_class, model_params,
                            criterion, optimizer_class, optimizer_params,
                            num_epochs=50, device='cpu', k_folds=5, batch_size=8,
                            scheduler_class=None, scheduler_params=None, use_emotion=False):
    """
    K折交叉验证训练多任务模型
    """
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    fold_results = []
    all_train_losses = []
    all_val_losses = []
    all_train_maes = []
    all_val_maes = []
    all_train_rmses = []
    all_val_rmses = []
    best_model_states = []
    
    X_data = X_ppg if X_prv is None else X_ppg
    
    for fold, (train_ids, val_ids) in enumerate(kfold.split(X_data)):
        print(f'\n正在训练第 {fold+1}/{k_folds} 折')
        print(f"  训练集大小: {len(train_ids)}, 验证集大小: {len(val_ids)}")
        
        model = model_class(**model_params)
        model.to(device)
        
        optimizer = optimizer_class(model.parameters(), **optimizer_params)
        
        scheduler = None
        if scheduler_class is not None:
            scheduler = scheduler_class(optimizer, **scheduler_params)
        
        X_ppg_train = X_ppg[train_ids]
        X_ppg_val = X_ppg[val_ids]
        y_stress_train = y_stress[train_ids]
        y_stress_val = y_stress[val_ids]
        
        if X_prv is not None:
            X_prv_train = X_prv[train_ids]
            X_prv_val = X_prv[val_ids]
        else:
            X_prv_train = X_ppg_train
            X_prv_val = X_ppg_val
        
        if use_emotion and y_emotion is not None:
            y_emotion_train = y_emotion[train_ids]
            y_emotion_val = y_emotion[val_ids]
            
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
                torch.tensor(y_stress_train, dtype=torch.float32).reshape(-1, 1)
            )
            val_dataset = TensorDataset(
                torch.tensor(X_ppg_val, dtype=torch.float32),
                torch.tensor(y_stress_val, dtype=torch.float32).reshape(-1, 1)
            )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        train_losses, val_losses, train_maes, val_maes, train_rmses, val_rmses, best_model_state = train_multi_task_model(
            model, train_loader, val_loader, criterion, optimizer, num_epochs, device, scheduler, use_emotion
        )
        
        best_model_states.append(best_model_state)
        
        model.load_state_dict(best_model_state)
        model.eval()
        
        final_val_loss = val_losses[-1]
        final_val_mae = val_maes[-1]
        final_val_rmse = val_rmses[-1]
        
        fold_results.append({
            'fold': fold + 1,
            'val_loss': final_val_loss,
            'val_mae': final_val_mae,
            'val_rmse': final_val_rmse
        })
        
        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)
        all_train_maes.append(train_maes)
        all_val_maes.append(val_maes)
        all_train_rmses.append(train_rmses)
        all_val_rmses.append(val_rmses)
        
        print(f'第 {fold+1} 折 - Val Loss: {final_val_loss:.4f}, MAE: {final_val_mae:.4f}, RMSE: {final_val_rmse:.4f}')
    
    avg_val_loss = np.mean([r['val_loss'] for r in fold_results])
    avg_val_mae = np.mean([r['val_mae'] for r in fold_results])
    avg_val_rmse = np.mean([r['val_rmse'] for r in fold_results])
    
    print(f"\nK折交叉验证结果:")
    print(f"平均验证损失: {avg_val_loss:.4f}")
    print(f"平均验证MAE: {avg_val_mae:.4f}")
    print(f"平均验证RMSE: {avg_val_rmse:.4f}")
    
    return fold_results, (all_train_losses, all_val_losses, all_train_maes, all_val_maes, all_train_rmses, all_val_rmses), best_model_states


def evaluate_multi_task_model(model, test_loader, criterion, device='cpu'):
    """评估多任务模型"""
    model.eval()
    test_loss = 0
    test_mae = 0
    test_rmse = 0
    predictions = []
    targets = []
    emotion_preds = []
    emotion_targets_list = []
    
    batch_count = 0
    with torch.no_grad():
        for batch_data in test_loader:
            if len(batch_data) == 4:
                ppg_data, prv_data, stress_target, emotion_target = batch_data
                ppg_data = ppg_data.to(device)
                prv_data = prv_data.to(device)
                stress_target = stress_target.to(device)
                emotion_target = emotion_target.to(device)
            else:
                ppg_data, stress_target = batch_data[0], batch_data[1]
                ppg_data = ppg_data.to(device)
                stress_target = stress_target.to(device)
                prv_data = None
                emotion_target = None
            
            if ppg_data.dim() == 2:
                ppg_data = ppg_data.unsqueeze(-1)
            if prv_data is not None and prv_data.dim() == 2:
                prv_data = prv_data.unsqueeze(-1)
            
            if hasattr(model, 'compute_loss'):
                output, emotion_pred = model(ppg_data, prv_data)
                if emotion_target is not None:
                    emotion_preds.extend(emotion_pred.argmax(dim=1).cpu().numpy())
                    emotion_targets_list.extend(emotion_target.cpu().numpy())
            else:
                output = model(ppg_data)
            
            loss = criterion(output.squeeze(), stress_target.squeeze())
            test_loss += loss.item()
            
            diff = torch.abs(output.squeeze() - stress_target.squeeze())
            mae = torch.mean(diff)
            rmse = torch.sqrt(torch.mean(diff ** 2))
            
            test_mae += mae.item()
            test_rmse += rmse.item()
            
            output_squeezed = output.squeeze()
            target_squeezed = stress_target.squeeze()
            
            if output_squeezed.dim() == 0:
                predictions.append(output_squeezed.item())
                targets.append(target_squeezed.item())
            else:
                predictions.extend(output_squeezed.cpu().numpy())
                targets.extend(target_squeezed.cpu().numpy())
            
            batch_count += 1
    
    test_loss /= batch_count
    test_mae /= batch_count
    test_rmse /= batch_count
    
    print(f'Test Loss: {test_loss:.4f}, MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}')
    
    if emotion_preds and emotion_targets_list:
        emotion_acc = np.mean(np.array(emotion_preds) == np.array(emotion_targets_list))
        print(f'Emotion Accuracy: {emotion_acc:.4f}')
    
    return test_loss, test_mae, test_rmse, predictions, targets