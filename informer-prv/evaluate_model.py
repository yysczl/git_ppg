import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 导入必要的模块
from data_processor import load_and_preprocess_data, PRVDataset, set_seed
from model_trainer import load_and_evaluate_model
from plot_utils import plot_predictions
from models import LSTM, GRU, BiLSTM, TemporalConvNet, TimeSeriesTransformer, Informer

# 设置随机种子以确保结果可复现
set_seed(42)

def main():
    # 文件路径
    file_path = "c:\\Users\\12991\\Desktop\\ppg-code\\regression_dataset\\PRV\\StressPRV.csv"
    model_path = "final_BiLSTM_model.pth"  # 根据实际保存的模型文件名进行修改
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"数据文件 {file_path} 不存在!")
        return
    
    if not os.path.exists(model_path):
        print(f"模型文件 {model_path} 不存在!")
        return
    
    # 加载和预处理数据
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = load_and_preprocess_data(file_path)
    X_concat = np.concatenate([X_test], axis=0)
    y_concat = np.concatenate([y_test], axis=0)
    
    print("数据加载完成!")
    print(f"测试集大小: {X_concat.shape}")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    batch_size = 8

    # 定义损失函数
    criterion = nn.MSELoss()
    
    # 创建测试数据加载器
    test_dataset = PRVDataset(X_concat, y_concat)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # 加载保存的模型并评估
    test_loss, test_mae, test_rmse, predictions, targets = load_and_evaluate_model(
        model_path, test_loader, criterion, device
    )
    
    print("\n模型评估结果:")
    print(f"测试集 - Loss: {test_loss:.4f}, MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}")
    
    # 绘制预测结果
    plot_predictions(predictions, targets)
    
    # 显示图形
    plt.show()

if __name__ == "__main__":
    main()