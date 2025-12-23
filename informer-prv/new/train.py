import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import KFold

from models import LSTM, GRU, BiLSTM, TemporalConvNet, TimeSeriesTransformer, Informer


# 训练模型
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, device='cpu', scheduler=None):
    train_losses = []
    val_losses = []
    train_maes = []
    val_maes = []
    train_rmses = []
    val_rmses = []
    
    # 将模型移至设备
    model.to(device)
    
    # 用于保存最佳模型
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_mae = 0
        train_rmse = 0
        
        batch_count = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            # 将数据移至设备
            data, target = data.to(device), target.to(device)
            
            # 清除梯度
            optimizer.zero_grad()
            
            # 前向传播
            # 检查模型类型以确定输入格式
            batch_size, seq_len = data.shape
            
            if hasattr(model, 'forward') and 'tgt' in model.forward.__code__.co_varnames:
                # 对于Informer模型，我们需要准备源序列和目标序列
                tgt = torch.zeros(batch_size, 1, 1).to(device)  # 目标序列，维度为[batch_size, 1, 1]
                # 重塑输入数据为[batch_size, seq_len, 1]以适应Informer模型
                data = data.unsqueeze(-1)
                # 前向传播
                output = model(data, tgt)
            else:
                # 对于其他模型，直接输入数据
                data = data.unsqueeze(-1)
                output = model(data)
            
            # 计算损失
            loss = criterion(output.squeeze(), target.squeeze())
            
            # 反向传播
            loss.backward()
            
            # 更新参数
            optimizer.step()
            
            # 累加损失
            train_loss += loss.item()
            
            # 计算MAE和RMSE
            diff = torch.abs(output.squeeze() - target.squeeze())
            mae = torch.mean(diff)
            rmse = torch.sqrt(torch.mean(diff ** 2))
            
            train_mae += mae.item()
            train_rmse += rmse.item()
            
            batch_count += 1
        
        # 计算平均损失
        train_loss /= batch_count
        train_mae /= batch_count
        train_rmse /= batch_count
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_mae = 0
        val_rmse = 0
        
        val_batch_count = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_loader):
                # 将数据移至设备
                data, target = data.to(device), target.to(device)
                
                # 前向传播
                batch_size, seq_len = data.shape
                
                if hasattr(model, 'forward') and 'tgt' in model.forward.__code__.co_varnames:
                    # 对于Informer模型
                    tgt = torch.zeros(batch_size, 1, 1).to(device)
                    data = data.unsqueeze(-1)
                    output = model(data, tgt)
                else:
                    # 对于其他模型
                    data = data.unsqueeze(-1)
                    output = model(data)
                
                # 计算损失
                loss = criterion(output.squeeze(), target.squeeze())
                
                # 累加损失
                val_loss += loss.item()
                
                # 计算MAE和RMSE
                diff = output.squeeze() - target.squeeze()
                mae = torch.mean(torch.abs(diff))
                rmse = torch.sqrt(torch.mean(diff ** 2))
                
                # 添加调试信息，检查MAE和RMSE计算（仅第一轮第一批次）
                if epoch == 0 and batch_idx == 0:
                    print(f"调试信息 - 第{epoch+1}轮, 第{batch_idx+1}批次:")
                    print(f"  输出值范围: {output.squeeze().min().item():.4f} ~ {output.squeeze().max().item():.4f}")
                    print(f"  目标值范围: {target.squeeze().min().item():.4f} ~ {target.squeeze().max().item():.4f}")
                    print(f"  差值范围: {diff.min().item():.4f} ~ {diff.max().item():.4f}")
                    print(f"  MAE: {mae.item():.4f}, RMSE: {rmse.item():.4f}")
                
                val_mae += mae.item()
                val_rmse += rmse.item()
                val_batch_count += 1
            
        # 计算平均损失
        val_loss /= val_batch_count
        val_mae /= val_batch_count
        val_rmse /= val_batch_count
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
        
        # 更新学习率
        if scheduler is not None:
            scheduler.step(val_loss)
        
        # 保存损失
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_maes.append(train_mae)
        val_maes.append(val_mae)
        train_rmses.append(train_rmse)
        val_rmses.append(val_rmse)
        
        # 打印进度（每10个epoch打印一次）
        if (epoch + 1) % 10 == 0:
            print(f'Epoch: {epoch+1}/{num_epochs}')
            print(f'Train Loss: {train_loss:.4f}, MAE: {train_mae:.4f}, RMSE: {train_rmse:.4f}')
            print(f'Val Loss: {val_loss:.4f}, MAE: {val_mae:.4f}, RMSE: {val_rmse:.4f}')
            # 打印当前学习率
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Current Learning Rate: {current_lr:.6f}')
    
    return train_losses, val_losses, train_maes, val_maes, train_rmses, val_rmses, best_model_state


# K折交叉验证训练模型
def train_model_kfold(X, y, model_class, model_params, criterion, optimizer_class, optimizer_params,
                      num_epochs=50, device='cpu', k_folds=5, batch_size=8, scheduler_class=None, scheduler_params=None):
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    fold_results = []
    all_train_losses = []
    all_val_losses = []
    all_train_maes = []
    all_val_maes = []
    all_train_rmses = []
    all_val_rmses = []
    best_model_states = []  # 保存每折的最佳模型权重

    for fold, (train_ids, val_ids) in enumerate(kfold.split(X)):
        print(f'正在训练第 {fold+1}/{k_folds} 折')

        # 添加调试信息，检查数据划分
        print(f"  训练集大小: {len(train_ids)}, 验证集大小: {len(val_ids)}")

        # 创建模型实例
        model = model_class(**model_params)
        model.to(device)

        # 创建优化器
        optimizer = optimizer_class(model.parameters(), **optimizer_params)

        # 创建学习率调度器
        scheduler = None
        if scheduler_class is not None:
            scheduler = scheduler_class(optimizer, **scheduler_params)

        # 采样数据
        X_train_fold = X[train_ids]
        y_train_fold = y[train_ids]
        X_val_fold = X[val_ids]
        y_val_fold = y[val_ids]

        # 添加调试信息，检查数据分布
        print(f"  训练目标值范围: {y_train_fold.min():.4f} ~ {y_train_fold.max():.4f}")
        print(f"  验证目标值范围: {y_val_fold.min():.4f} ~ {y_val_fold.max():.4f}")

        # 创建数据集和数据加载器
        train_dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_train_fold, dtype=torch.float32),
            torch.tensor(y_train_fold, dtype=torch.float32).reshape(-1, 1)
        )
        val_dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_val_fold, dtype=torch.float32),
            torch.tensor(y_val_fold, dtype=torch.float32).reshape(-1, 1)
        )

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

        # 训练模型
        train_losses, val_losses, train_maes, val_maes, train_rmses, val_rmses, best_model_state = train_model(
            model, train_loader, val_loader, criterion, optimizer, num_epochs, device, scheduler
        )

        # 保存最佳模型权重
        best_model_states.append(best_model_state)

        # 评估当前折的最终性能
        model.eval()
        val_loss = 0
        val_mae = 0
        val_rmse = 0

        val_batch_count = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                batch_size, seq_len = data.shape
                
                if hasattr(model, 'forward') and 'tgt' in model.forward.__code__.co_varnames:
                    # 对于Informer模型
                    tgt = torch.zeros(batch_size, 1, 1).to(device)
                    data = data.unsqueeze(-1)
                    output = model(data, tgt)
                else:
                    # 对于其他模型
                    data = data.unsqueeze(-1)
                    output = model(data)

                loss = criterion(output.squeeze(), target.squeeze())
                val_loss += loss.item()

                # 计算MAE和RMSE
                diff = torch.abs(output.squeeze() - target.squeeze())
                mae = torch.mean(diff)
                rmse = torch.sqrt(torch.mean(diff ** 2))

                val_mae += mae.item()
                val_rmse += rmse.item()

                val_batch_count += 1

            val_loss /= val_batch_count
            val_mae /= val_batch_count
            val_rmse /= val_batch_count

        fold_results.append({
            'fold': fold + 1,
            'val_loss': val_loss,
            'val_mae': val_mae,
            'val_rmse': val_rmse
        })

        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)
        all_train_maes.append(train_maes)
        all_val_maes.append(val_maes)
        all_train_rmses.append(train_rmses)
        all_val_rmses.append(val_rmses)

        print(f'第 {fold+1} 折 - Val Loss: {val_loss:.4f}, MAE: {val_mae:.4f}, RMSE: {val_rmse:.4f}')

    # 计算平均结果
    avg_val_loss = np.mean([r['val_loss'] for r in fold_results])
    avg_val_mae = np.mean([r['val_mae'] for r in fold_results])
    avg_val_rmse = np.mean([r['val_rmse'] for r in fold_results])

    print(f"\nK折交叉验证结果:")
    print(f"平均验证损失: {avg_val_loss:.4f}")
    print(f"平均验证MAE: {avg_val_mae:.4f}")
    print(f"平均验证RMSE: {avg_val_rmse:.4f}")

    for r in fold_results:
        print(f"第 {r['fold']} 折 - Loss: {r['val_loss']:.4f}, MAE: {r['val_mae']:.4f}, RMSE: {r['val_rmse']:.4f}")

    return fold_results, (all_train_losses, all_val_losses, all_train_maes, all_val_maes, all_train_rmses, all_val_rmses), best_model_states


def evaluate_model(model, test_loader, criterion, device='cpu'):
    model.eval()
    test_loss = 0
    test_mae = 0
    test_rmse = 0
    predictions = []
    targets = []
    
    batch_count = 0
    with torch.no_grad():
        for data, target in test_loader:
            # 将数据移至设备
            data, target = data.to(device), target.to(device)
            
            # 前向传播
            batch_size, seq_len = data.shape
            
            if hasattr(model, 'forward') and 'tgt' in model.forward.__code__.co_varnames:
                # 对于Informer模型
                tgt = torch.zeros(batch_size, 1, 1).to(device)
                data = data.unsqueeze(-1)
                output = model(data, tgt)
            else:
                # 对于其他模型
                data = data.unsqueeze(-1)
                output = model(data)
            
            # 计算损失
            loss = criterion(output.squeeze(), target.squeeze())
            
            # 累加损失
            test_loss += loss.item()
            
            # 计算MAE和RMSE
            diff = torch.abs(output.squeeze() - target.squeeze())
            mae = torch.mean(diff)
            rmse = torch.sqrt(torch.mean(diff ** 2))
            
            test_mae += mae.item()
            test_rmse += rmse.item()
            
            # 保存预测和目标
            # 处理output.squeeze()可能为0维张量的情况
            output_squeezed = output.squeeze()
            target_squeezed = target.squeeze()
            
            if output_squeezed.dim() == 0:  # 如果是0维张量
                predictions.append(output_squeezed.item())
                targets.append(target_squeezed.item())
            else:  # 如果是1维张量
                predictions.extend(output_squeezed.cpu().numpy())
                targets.extend(target_squeezed.cpu().numpy())
            
            batch_count += 1
    
    # 计算平均损失
    test_loss /= batch_count
    test_mae /= batch_count
    test_rmse /= batch_count
    
    print(f'Test Loss: {test_loss:.4f}, MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}')
    
    return test_loss, test_mae, test_rmse, predictions, targets


# 训练最终模型（用于部署）
def train_final_model(X, y, model_class, model_params, criterion, optimizer_class, optimizer_params,
                      num_epochs=50, device='cpu', batch_size=8, scheduler_class=None, scheduler_params=None):
    """
    在完整数据集上训练最终模型，用于部署
    """
    print("开始训练最终模型...")
    
    # 创建模型实例
    model = model_class(**model_params)
    model.to(device)
    
    # 创建优化器
    optimizer = optimizer_class(model.parameters(), **optimizer_params)
    
    # 创建学习率调度器
    scheduler = None
    if scheduler_class is not None:
        scheduler = scheduler_class(optimizer, **scheduler_params)
    
    # 创建数据集和数据加载器
    dataset = torch.utils.data.TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
    )
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 记录训练过程
    train_losses = []
    train_maes = []
    train_rmses = []
    
    # 用于保存最佳模型
    best_train_loss = float('inf')
    best_model_state = None
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_mae = 0
        train_rmse = 0
        
        batch_count = 0
        for batch_idx, (data, target) in enumerate(data_loader):
            # 将数据移至设备
            data, target = data.to(device), target.to(device)
            
            # 清除梯度
            optimizer.zero_grad()
            
            # 前向传播
            batch_size_batch, seq_len = data.shape
            
            if hasattr(model, 'forward') and 'tgt' in model.forward.__code__.co_varnames:
                # 对于Informer模型，我们需要准备源序列和目标序列
                tgt = torch.zeros(batch_size_batch, 1, 1).to(device)  # 目标序列，维度为[batch_size, 1, 1]
                # 重塑输入数据为[batch_size, seq_len, 1]以适应Informer模型
                data = data.unsqueeze(-1)
                # 前向传播
                output = model(data, tgt)
            else:
                # 对于其他模型，直接输入数据
                data = data.unsqueeze(-1)
                output = model(data)
            
            # 计算损失
            loss = criterion(output.squeeze(), target.squeeze())
            
            # 反向传播
            loss.backward()
            
            # 更新参数
            optimizer.step()
            
            # 累加损失
            train_loss += loss.item()
            
            # 计算MAE和RMSE
            diff = torch.abs(output.squeeze() - target.squeeze())
            mae = torch.mean(diff)
            rmse = torch.sqrt(torch.mean(diff ** 2))
            
            train_mae += mae.item()
            train_rmse += rmse.item()
            
            batch_count += 1
        
        # 计算平均损失
        train_loss /= batch_count
        train_mae /= batch_count
        train_rmse /= batch_count
        
        # 保存最佳模型
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            best_model_state = model.state_dict().copy()
        
        # 更新学习率
        if scheduler is not None:
            scheduler.step(train_loss)
        
        # 保存损失
        train_losses.append(train_loss)
        train_maes.append(train_mae)
        train_rmses.append(train_rmse)
        
        # 打印进度（每10个epoch打印一次）
        if (epoch + 1) % 10 == 0:
            print(f'Epoch: {epoch+1}/{num_epochs}')
            print(f'Train Loss: {train_loss:.4f}, MAE: {train_mae:.4f}, RMSE: {train_rmse:.4f}')
            # 打印当前学习率
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Current Learning Rate: {current_lr:.6f}')
    
    # 加载最佳模型权重
    model.load_state_dict(best_model_state)
    
    print(f"最终模型训练完成! 最佳训练损失: {best_train_loss:.4f}")
    
    return model, train_losses, train_maes, train_rmses


def load_and_evaluate_model(model_path, test_loader, criterion, device='cpu'):
    """
    加载已保存的模型并对测试集进行评估
    
    Args:
        model_path (str): 模型文件路径
        test_loader: 测试数据加载器
        criterion: 损失函数
        device (str): 设备类型
    
    Returns:
        tuple: (test_loss, test_mae, test_rmse, predictions, targets)
    """
    print(f"正在从 {model_path} 加载模型...")
    
    # 直接加载完整模型
    # 添加对自定义模型类的安全声明
    torch.serialization.add_safe_globals([Informer, LSTM, GRU, BiLSTM, TemporalConvNet, TimeSeriesTransformer])
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.to(device)
    
    print("模型加载完成，开始评估...")
    
    # 使用现有的evaluate_model函数进行评估
    test_loss, test_mae, test_rmse, predictions, targets = evaluate_model(
        model, test_loader, criterion, device
    )
    
    print("模型评估完成!")
    return test_loss, test_mae, test_rmse, predictions, targets
