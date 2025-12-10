"""
PPG-Former-DualStream 主程序
融合多尺度时频Transformer与双流协同的多任务心理压力预测
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from data_processor import load_and_preprocess_data, load_and_preprocess_prv_data, set_seed
from plot_utils import plot_training_process, plot_predictions
from ppg_former_model import PPGFormer
from dual_stream_model import PPGFormerDualStream, DualStreamOnly
from multi_task_trainer import (
    train_kfold_multi_task,
    evaluate_multi_task_model
)

set_seed(42)


def main():
    ppg_file_path = "../dataset/PPG/StressPPG.csv"
    prv_file_path = "../dataset/PRV/StressPRV.csv"
    
    print("=" * 60)
    print("PPG-Former-DualStream 多任务压力预测模型")
    print("=" * 60)
    
    use_dual_stream = True
    use_ppg_only = False
    
    if use_dual_stream and os.path.exists(ppg_file_path) and os.path.exists(prv_file_path):
        print("\n加载PPG和PRV双模态数据...")
        X_ppg_train, y_ppg_train, X_ppg_val, y_ppg_val, X_ppg_test, y_ppg_test, ppg_scaler = load_and_preprocess_data(ppg_file_path)
        X_prv_train, y_prv_train, X_prv_val, y_prv_val, X_prv_test, y_prv_test, prv_scaler = load_and_preprocess_prv_data(prv_file_path)
        
        X_ppg_combined = np.concatenate([X_ppg_train, X_ppg_val], axis=0)
        X_prv_combined = np.concatenate([X_prv_train, X_prv_val], axis=0)
        y_combined = np.concatenate([y_ppg_train, y_ppg_val], axis=0)
        
        model_type = "PPGFormerDualStream"
        model_class = PPGFormerDualStream
        model_params = {
            'ppg_input_dim': 1,
            'prv_input_dim': 1,
            'd_model': 128,
            'n_heads': 8,
            'd_ff': 512,
            'ppg_layers': 3,
            'prv_layers': 2,
            'fusion_layers': 2,
            'num_emotions': 5,
            'scales': [1, 3, 5, 7],
            'dropout': 0.1
        }
        
        X_test = X_ppg_test
        y_test = y_ppg_test
        X_prv_test_data = X_prv_test
        
    elif os.path.exists(ppg_file_path):
        print("\n仅加载PPG数据，使用PPG-Former模型...")
        X_train, y_train, X_val, y_val, X_test, y_test, scaler = load_and_preprocess_data(ppg_file_path)
        
        X_ppg_combined = np.concatenate([X_train, X_val], axis=0)
        X_prv_combined = None
        y_combined = np.concatenate([y_train, y_val], axis=0)
        
        model_type = "PPGFormer"
        model_class = PPGFormer
        model_params = {
            'input_dim': 1,
            'output_dim': 1,
            'd_model': 128,
            'n_heads': 8,
            'd_ff': 512,
            'num_layers': 3,
            'scales': [1, 3, 5, 7],
            'dropout': 0.1
        }
        X_prv_test_data = None
        
    elif os.path.exists(prv_file_path):
        print("\n仅加载PRV数据，使用DualStreamOnly模型...")
        X_train, y_train, X_val, y_val, X_test, y_test, scaler = load_and_preprocess_prv_data(prv_file_path)
        
        X_ppg_combined = np.concatenate([X_train, X_val], axis=0)
        X_prv_combined = None
        y_combined = np.concatenate([y_train, y_val], axis=0)
        
        model_type = "DualStreamOnly"
        model_class = DualStreamOnly
        model_params = {
            'input_dim': 1,
            'output_dim': 1,
            'd_model': 128,
            'n_heads': 8,
            'num_layers': 3,
            'dropout': 0.1
        }
        X_prv_test_data = None
        
    else:
        print("数据文件不存在，请检查路径！")
        return
    
    print(f"\n使用模型: {model_type}")
    print(f"PPG数据形状: {X_ppg_combined.shape}")
    if X_prv_combined is not None:
        print(f"PRV数据形状: {X_prv_combined.shape}")
    print(f"标签形状: {y_combined.shape}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    criterion = nn.MSELoss()
    
    optimizer_class = optim.AdamW
    optimizer_params = {
        'lr': 0.0005,
        'weight_decay': 1e-4
    }
    
    scheduler_class = optim.lr_scheduler.ReduceLROnPlateau
    scheduler_params = {
        'mode': 'min',
        'factor': 0.5,
        'patience': 20
    }
    
    k_folds = 5
    num_epochs = 300
    batch_size = 8
    use_emotion = False
    
    print(f"\n开始{k_folds}折交叉验证训练...")
    
    fold_results, all_losses, best_model_states = train_kfold_multi_task(
        X_ppg_combined, X_prv_combined, y_combined, None,
        model_class, model_params,
        criterion,
        optimizer_class, optimizer_params,
        num_epochs, device,
        k_folds, batch_size,
        scheduler_class, scheduler_params,
        use_emotion
    )
    
    avg_train_losses = np.mean(all_losses[0], axis=0)
    avg_val_losses = np.mean(all_losses[1], axis=0)
    avg_train_maes = np.mean(all_losses[2], axis=0)
    avg_val_maes = np.mean(all_losses[3], axis=0)
    avg_train_rmses = np.mean(all_losses[4], axis=0)
    avg_val_rmses = np.mean(all_losses[5], axis=0)
    
    plot_training_process(avg_train_losses, avg_val_losses, avg_train_maes, avg_val_maes, 
                         avg_train_rmses, avg_val_rmses, model_type, "ppg_former")
    
    best_fold_index = np.argmin([result['val_loss'] for result in fold_results])
    print(f"\n选择第 {best_fold_index+1} 折的模型作为最佳模型")
    
    best_model = model_class(**model_params)
    best_model.load_state_dict(best_model_states[best_fold_index])
    best_model.to(device)
    
    if X_prv_test_data is not None:
        test_dataset = TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(X_prv_test_data, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)
        )
    else:
        test_dataset = TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)
        )
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    print("\n在测试集上评估模型...")
    test_loss, test_mae, test_rmse, predictions, targets = evaluate_multi_task_model(
        best_model, test_loader, criterion, device
    )
    
    plot_predictions(predictions, targets, model_type, "ppg_former")
    
    print("\n" + "=" * 60)
    print("模型训练和评估完成!")
    print(f"最终测试结果 - Loss: {test_loss:.4f}, MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}")
    print("=" * 60)
    
    model_save_path = f"{model_type}_model.pth"
    torch.save(best_model, model_save_path)
    print(f"\n最佳模型已保存到: {model_save_path}")
    
    total_params = sum(p.numel() for p in best_model.parameters())
    trainable_params = sum(p.numel() for p in best_model.parameters() if p.requires_grad)
    print(f"模型总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")


if __name__ == "__main__":
    main()