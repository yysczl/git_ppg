import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# 导入拆分后的模块
from data_processor import load_and_preprocess_data, PRVDataset, set_seed
from model_trainer import evaluate_model, train_model_kfold, train_final_model, load_and_evaluate_model
from plot_utils import plot_training_process, plot_predictions
# from informer_model import Informer
from models import LSTM, GRU, BiLSTM, TemporalConvNet, TimeSeriesTransformer, Informer

# 设置随机种子以确保结果可复现
set_seed(42)

# 主函数
def main():
    # 文件路径
    file_path = "../dataset/PRV/StressPRV.csv"
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"文件 {file_path} 不存在!")
        return
    
    # 加载和预处理数据
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = load_and_preprocess_data(file_path)
    
    # 合并训练集和验证集用于k折交叉验证
    X_combined = np.concatenate([X_train, X_val], axis=0)
    y_combined = np.concatenate([y_train, y_val], axis=0)
    
    print("数据加载完成!")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 定义模型类型和参数
    model_type = "LSTM"  # 可选: "Informer", "LSTM", "GRU", "BiLSTM", "TCN", "Transformer"
    
    # 根据模型类型设置模型参数
    if model_type == "Informer":
        model_class = Informer
        model_params = {
            'input_dim': 1,  # 输入维度为1，因为我们将每个时间点的值作为一个特征
            'output_dim': 1,  # 输出维度为1，因为我们预测一个标量值
            'd_model': 128,   # 增加模型维度
            'n_heads': 8,     # 增加注意力头数
            'd_ff': 512,      # 增加前馈网络维度
            'enc_layers': 3,  # 增加编码器层数
            'dec_layers': 2,  # 增加解码器层数
            'dropout': 0.1    # Dropout率
        }
    elif model_type == "LSTM":
        model_class = LSTM
        model_params = {
            'input_dim': 1,
            'hidden_dim': 128,
            'num_layers': 2,
            'output_dim': 1
        }
    elif model_type == "GRU":
        model_class = GRU
        model_params = {
            'input_dim': 1,
            'hidden_dim': 128,
            'num_layers': 2,
            'output_dim': 1
        }
    elif model_type == "BiLSTM":
        model_class = BiLSTM
        model_params = {
            'input_dim': 1,
            'hidden_dim': 128,
            'num_layers': 2,
            'output_dim': 1
        }
    elif model_type == "TCN":
        model_class = TemporalConvNet
        model_params = {
            'input_dim': 1,
            'output_dim': 1,
            'num_channels': [64, 64, 128, 128],  # TCN层的通道数
            'kernel_size': 3,
            'dropout': 0.1
        }
    elif model_type == "Transformer":
        model_class = TimeSeriesTransformer
        model_params = {
            'input_dim': 1,
            'num_heads': 8,
            'num_layers': 3,
            'output_dim': 1,
            'hidden_dim': 128,
            'dropout_rate': 0.1
        }
    
    # 定义损失函数
    criterion = nn.MSELoss()  # 均方误差损失
    
    # 定义优化器参数
    optimizer_class = optim.AdamW
    optimizer_params = {
        'lr': 0.0005,
        'weight_decay': 1e-4
    }
    
    # 定义学习率调度器参数
    scheduler_class = optim.lr_scheduler.ReduceLROnPlateau
    scheduler_params = {
        'mode': 'min',
        'factor': 0.5,
        'patience': 20
    }
    
    # K折交叉验证训练
    k_folds = 5
    num_epochs = 100
    batch_size = 8
    
    fold_results, all_losses, best_model_states = train_model_kfold(
        X_combined, y_combined,
        model_class, model_params,
        criterion,
        optimizer_class, optimizer_params,
        num_epochs, device,
        k_folds, batch_size,
        scheduler_class, scheduler_params
    )
    
    # 绘制最后一折的训练过程
    # 计算所有折的平均损失
    avg_train_losses = np.mean(all_losses[0], axis=0)
    avg_val_losses = np.mean(all_losses[1], axis=0)
    avg_train_maes = np.mean(all_losses[2], axis=0)
    avg_val_maes = np.mean(all_losses[3], axis=0)
    avg_train_rmses = np.mean(all_losses[4], axis=0)
    avg_val_rmses = np.mean(all_losses[5], axis=0)
    
    plot_training_process(avg_train_losses, avg_val_losses, avg_train_maes, avg_val_maes, avg_train_rmses, avg_val_rmses, model_type, "prv")

    # 使用所有折的模型进行集成预测，而不是选择单折中表现最好的模型
    print("使用所有折的模型进行集成预测...")
    # 创建测试数据加载器
    test_dataset = PRVDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # 加载所有折的模型
    ensemble_models = []
    for i, model_state in enumerate(best_model_states):
        model = model_class(**model_params)
        model.load_state_dict(model_state)
        model.to(device)
        model.eval()
        ensemble_models.append(model)
    
    # 评估集成模型
    ensemble_predictions = []
    targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            batch_size, seq_len = data.shape
            
            # 收集所有模型的预测结果
            fold_predictions = []
            for model in ensemble_models:
                if hasattr(model, 'forward') and 'tgt' in model.forward.__code__.co_varnames:
                    # 对于Informer模型
                    tgt = torch.zeros(batch_size, 1, 1).to(device)
                    data_input = data.unsqueeze(-1)
                    output = model(data_input, tgt)
                else:
                    # 对于其他模型
                    data_input = data.unsqueeze(-1)
                    output = model(data_input)
                
                fold_predictions.append(output.squeeze())
            
            # 计算所有模型预测的平均值
            avg_prediction = torch.mean(torch.stack(fold_predictions), dim=0)
            
            # 处理可能的0维张量情况
            if avg_prediction.dim() == 0:  # 如果是0维张量
                ensemble_predictions.append(avg_prediction.item())
                targets.append(target.squeeze().item())
            else:  # 如果是1维张量
                ensemble_predictions.extend(avg_prediction.cpu().numpy())
                targets.extend(target.squeeze().cpu().numpy())
    
    # 计算集成模型的评估指标
    ensemble_predictions = np.array(ensemble_predictions)
    targets = np.array(targets)
    
    test_loss = np.mean((ensemble_predictions - targets) ** 2)
    test_mae = np.mean(np.abs(ensemble_predictions - targets))
    test_rmse = np.sqrt(test_loss)
    
    # 绘制预测结果
    plot_predictions(ensemble_predictions, targets, model_type, "prv")
    
    print("模型训练和评估完成!")
    print(f"测试集 - Loss: {test_loss:.4f}, MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}")
    
    # 训练用于部署的最终模型
    print("\n开始训练用于部署的最终模型...")
    
    # 合并所有数据用于最终训练 (训练集 + 验证集 + 测试集)
    X_full = np.concatenate([X_train, X_val], axis=0)
    y_full = np.concatenate([y_train, y_val], axis=0)
    
    print(f"完整数据集大小: {X_full.shape}")
    
    # 使用与交叉验证相同的超参数训练最终模型
    final_model, final_train_losses, final_train_maes, final_train_rmses = train_final_model(
        X_full, y_full,
        model_class, model_params,
        criterion,
        optimizer_class, optimizer_params,
        num_epochs, device,
        batch_size,
        scheduler_class, scheduler_params
    )
    
    # 保存最终模型
    model_save_path = f"prv_{model_type}_model.pth"
    torch.save(final_model, model_save_path)
    print(f"最终模型已保存到: {model_save_path}")
    
    # 示例：加载保存的模型并重新评估测试集
    print("\n示例：加载保存的模型并重新评估测试集...")
    loaded_test_loss, loaded_test_mae, loaded_test_rmse, loaded_predictions, loaded_targets = load_and_evaluate_model(
        model_save_path, test_loader, criterion, device
    )
    
    print("加载模型评估结果:")
    print(f"测试集 - Loss: {loaded_test_loss:.4f}, MAE: {loaded_test_mae:.4f}, RMSE: {loaded_test_rmse:.4f}")

if __name__ == "__main__":
    main()