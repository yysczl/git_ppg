import numpy as np
import pandas as pd
import torch
import random
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# 设置随机种子以确保结果可复现
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 加载数据
def load_and_preprocess_data(file_path, test_size=0.1, val_size=0.1):
    # 读取CSV文件，跳过表头
    try:
        # 首先尝试读取前几行来检查是否有表头
        sample = pd.read_csv(file_path, header=None, nrows=5, skiprows=1)
        print("读取的前5行数据样本:")
        print(sample.head())
        
        # 跳过第一行
        data = pd.read_csv(file_path, header=None, skiprows=1)

    except Exception as e:
        print(f"读取CSV文件时出错: {e}")
        print("尝试使用不同的编码方式读取...")
        # 尝试使用不同的编码方式
        data = pd.read_csv(file_path, header=None, encoding='latin1')
    
    # 检查数据形状
    print(f"数据形状: {data.shape}")
    
    # 提取特征和标签
    # 假设前1800列为特征，最后一列为标签
    features = data.iloc[:, :1800].values
    labels = data.iloc[:, -1].values
    
    print(f"特征形状: {features.shape}, 标签形状: {labels.shape}")
    
    # 标准化特征
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # 划分训练集、验证集和测试集
    # 首先划分训练集和临时测试集
    X_train, X_temp, y_train, y_temp = train_test_split(
        features_scaled, labels, test_size=test_size+val_size, random_state=42
    )
    
    # 然后从临时测试集中划分验证集和测试集
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=test_size/(test_size+val_size), random_state=42
    )
    
    print(f"训练集: {X_train.shape}, 验证集: {X_val.shape}, 测试集: {X_test.shape}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, scaler


# 加载和预处理PRV数据
def load_and_preprocess_prv_data(file_path, test_size=0.1, val_size=0.1):
    """
    加载和预处理PRV数据
    PRV数据特点:
    - 第一行为表头
    - 共有80列特征数据
    - 最后一列为压力值标签
    - 除去表头，共有90行样本
    """
    try:
        # 读取PRV数据，第一行为表头
        data = pd.read_csv(file_path)
    except Exception as e:
        print(f"读取PRV CSV文件时出错: {e}")
        print("尝试使用不同的编码方式读取...")
        # 尝试使用不同的编码方式
        data = pd.read_csv(file_path, encoding='latin1')
    
    # 检查数据形状
    print(f"PRV数据形状: {data.shape}")
    
    # 确认数据维度是否符合预期 (90行样本 + 1行表头) x (80列特征 + 1列标签)
    if data.shape[0] != 91 or data.shape[1] != 81:
        print(f"警告: PRV数据维度与预期不符。预期: 91行 x 81列, 实际: {data.shape[0]}行 x {data.shape[1]}列")
    
    # 提取特征和标签
    # 前80列为特征，最后一列为标签
    features = data.iloc[:, :80].values
    labels = data.iloc[:, -1].values
    
    print(f"PRV特征形状: {features.shape}, 标签形状: {labels.shape}")
    
    # 标准化特征
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # 划分训练集、验证集和测试集
    # 首先划分训练集和临时测试集
    X_train, X_temp, y_train, y_temp = train_test_split(
        features_scaled, labels, test_size=test_size+val_size, random_state=42
    )
    
    # 然后从临时测试集中划分验证集和测试集
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=test_size/(test_size+val_size), random_state=42
    )
    
    print(f"PRV训练集: {X_train.shape}, 验证集: {X_val.shape}, 测试集: {X_test.shape}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, scaler


# 加载和预处理瞳孔波数据
def load_and_preprocess_pupil_data(file_path, test_size=0.1, val_size=0.1):
    """
    加载和预处理瞳孔波数据
    瞳孔波数据特点:
    - 第一行为表头
    - 前5400列为特征数据（列名从1到5400）
    - 倒数第3列为"stress"标签
    - 共有164行（1行表头+163个样本）
    """
    try:
        # 读取瞳孔波数据，第一行为表头
        data = pd.read_csv(file_path)
        print("读取的瞳孔波数据样本:")
        print(data.head())
    except Exception as e:
        print(f"读取瞳孔波CSV文件时出错: {e}")
        print("尝试使用不同的编码方式读取...")
        # 尝试使用不同的编码方式
        data = pd.read_csv(file_path, encoding='latin1')
    
    # 检查数据形状
    print(f"瞳孔波数据形状: {data.shape}")
    
    # 确认数据维度是否符合预期 (163个样本 + 1行表头) x (5400列特征 + 一些标签列)
    if data.shape[0] != 163:
        print(f"警告: 瞳孔波数据行数与预期不符。预期: 164行, 实际: {data.shape[0]}行")
    
    # 提取特征和标签
    # 前5400列为特征
    features = data.iloc[:, :5400].values

    # 倒数第3列为stress标签
    labels = data.iloc[:, -3].values
    
    print(f"瞳孔波特征形状: {features.shape}, 标签形状: {labels.shape}")
    
    # 标准化特征
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # 划分训练集、验证集和测试集
    # 首先划分训练集和临时测试集
    X_train, X_temp, y_train, y_temp = train_test_split(
        features_scaled, labels, test_size=test_size+val_size, random_state=42
    )
    
    # 然后从临时测试集中划分验证集和测试集
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=test_size/(test_size+val_size), random_state=42
    )
    
    print(f"瞳孔波训练集: {X_train.shape}, 验证集: {X_val.shape}, 测试集: {X_test.shape}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, scaler
