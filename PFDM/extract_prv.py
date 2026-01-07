"""
从PPG信号中提取PRV（脉率变异性）数据
参考extract_maibo.ipynb中的PRV提取算法

| 步骤 | extract_maibo.ipynb (PRV函数) |
|------|------------------------------|
| 1. 滤波 | `apply_filtering(PPG)` 带通滤波 0.5-10Hz |
| 2. 峰值检测 | `systolic_peaks_1(fil)` distance=18 |
| 3. PP间期计算 | 相邻峰值差 |
| 4. 异常值处理 | `rate=len(PPG)/90`, 超范围用邻均值替代 |
| 5. 归一化 | `np.divide(PP_list1, len(PPG)/40)` |
| 6. 长度调整 | 不足80用均值填充，超过截取前80 |
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks
from itertools import chain
import os


def sampling_freq(signal, time=90):
    """计算采样频率"""
    return int(len(signal) / time)


def butter_bandpass(lowcut, highcut, fs, order=4):
    """巴特沃斯带通滤波器"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    # 防止high取到1，因为范围是（0，1）0和1都不可取到
    high = min(highcut, 0.999) / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def apply_filtering(signal):
    """应用滤波"""
    # 采样频率
    fs = sampling_freq(signal)
    print(f"采样频率: {fs} Hz")
    b, a = butter_bandpass(0.5, 10, fs, order=4)
    return filtfilt(b, a, signal)


def systolic_peaks_1(signal):
    """检测收缩峰值"""
    return find_peaks(signal, distance=18)[0]


def tfn_points_1(signal):
    """检测谷值点"""
    # 在这里，我使用还原信号，并获得高于0的峰值
    return find_peaks(signal * (-1), height=0, distance=25)[0]


def extract_PRV(PPG):
    """
    提取脉率变异性
    
    参数:
        PPG: PPG信号数组
    
    返回:
        PRV: 长度为80的PRV数组
    """
    PRV = []
    
    # 可视化点检测
    fil = apply_filtering(PPG)
    systolics = systolic_peaks_1(fil)
    tfns = tfn_points_1(fil)
    
    # 计算峰峰间期（PP interval）
    PP_list1 = []
    cnt = 0
    while (cnt < (len(systolics) - 1)):
        PP_interval = (systolics[cnt + 1] - systolics[cnt])  # 以样本数计算节拍之间的距离
        PP_list1.append(PP_interval)  # 附加到列表
        cnt += 1
    
    # 异常值处理
    cnt = 1
    rate = len(PPG) / 90
    c1 = rate * 0.5
    while (cnt < len(PP_list1) - 1):
        if PP_list1[cnt] > rate or PP_list1[cnt] < c1:
            PP_list1[cnt] = (PP_list1[cnt - 1] + PP_list1[cnt + 1]) / 2
        cnt = cnt + 1
    
    # 归一化
    num = len(PPG) / 40
    new_lst1 = np.divide(PP_list1, num)
    
    # 调整长度到80
    if len(new_lst1) < 80:
        # 如果长度不够80，用均值填充
        mean_value = np.mean(new_lst1)
        # 创建一个迭代器，它首先产生 new_lst1 的所有元素，然后是 mean_value 的重复
        padded_iter = chain(new_lst1, [mean_value] * (80 - len(new_lst1)))
        # 将迭代器转换为列表
        new_lst2 = list(padded_iter)
    else:
        new_lst2 = new_lst1[0:80]
    
    PRV.append(new_lst2)
    return np.array(PRV).reshape(-1)


def process_ppg_file(ppg_file_path, output_file_path=None, label_name=None):
    """
    处理PPG文件，提取PRV数据并保存
    
    参数:
        ppg_file_path: PPG数据文件路径
        output_file_path: 输出PRV文件路径，如果为None则自动生成
        label_name: PRV数据标签名称，如果为None则从文件名提取（用于表头）
    """
    print(f"正在处理文件: {ppg_file_path}")
    
    # 读取PPG数据（第一行是列标题）
    ppg_data = pd.read_csv(ppg_file_path, header=0)
    print(f"PPG数据形状: {ppg_data.shape}")
    
    # 获取PPG数据的标签列（最后一列）
    label_column_name = ppg_data.columns[-1]
    labels = ppg_data[label_column_name].values
    
    # 获取PPG信号数据（除去最后一列标签）
    ppg_signal_data = ppg_data.iloc[:, :-1]
    
    # 如果没有指定输出文件路径，自动生成
    if output_file_path is None:
        # 从PPG文件路径生成PRV文件路径
        ppg_dir = os.path.dirname(ppg_file_path)
        ppg_filename = os.path.basename(ppg_file_path)
        # 将PPG目录替换为PRV目录
        prv_dir = ppg_dir.replace('/PPG/', '/PRV/')
        output_file_path = os.path.join(prv_dir, ppg_filename)
    
    # 如果没有指定标签名称，从PPG文件的标签列获取
    if label_name is None:
        label_name = label_column_name
    
    # 处理每一行PPG数据，提取PRV
    prv_list = []
    for idx, row in ppg_signal_data.iterrows():
        print(f"处理第 {idx + 1}/{len(ppg_signal_data)} 行...")
        # 获取当前行的PPG信号，并转换为float类型
        ppg_signal = row.values.astype(float)
        
        try:
            # 提取PRV
            prv = extract_PRV(ppg_signal)
            prv_list.append(prv)
        except Exception as e:
            print(f"警告: 处理第 {idx + 1} 行时出错: {e}")
            # 如果出错，使用空值或前一行数据
            if len(prv_list) > 0:
                prv_list.append(prv_list[-1])
            else:
                prv_list.append(np.zeros(80))
    
    # 转换为DataFrame，并设置列名从1到80
    prv_df = pd.DataFrame(prv_list, columns=[str(i) for i in range(1, 81)])
    
    # 添加标签列（使用原始PPG数据的标签值）
    prv_df[label_name] = labels
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    
    # 保存PRV数据，包含表头，保留8位小数
    prv_df.to_csv(output_file_path, index=False, header=True, float_format='%.8f')
    print(f"PRV数据已保存到: {output_file_path}")
    print(f"PRV数据形状: {prv_df.shape}")


def batch_process_ppg_files(ppg_dir, prv_dir=None):
    """
    批量处理PPG文件夹中的所有CSV文件
    
    参数:
        ppg_dir: PPG数据文件夹路径
        prv_dir: PRV输出文件夹路径，如果为None则自动生成
    """
    if prv_dir is None:
        prv_dir = ppg_dir.replace('/PPG/', '/PRV/')
    
    # 确保输出目录存在
    os.makedirs(prv_dir, exist_ok=True)
    
    # 遍历PPG文件夹中的所有CSV文件
    for filename in os.listdir(ppg_dir):
        if filename.endswith('.csv'):
            ppg_file_path = os.path.join(ppg_dir, filename)
            prv_file_path = os.path.join(prv_dir, filename)
            
            try:
                process_ppg_file(ppg_file_path, prv_file_path)
            except Exception as e:
                print(f"错误: 处理文件 {filename} 时出错: {e}")
                continue


if __name__ == "__main__":
    # 示例：处理单个文件
    ppg_file = "/home/czl/git_ppg/PFDM/dataset/PPG/Anxiety.csv"
    prv_file = "/home/czl/git_ppg/PFDM/dataset/PRV/Anxiety_extracted.csv"
    
    # 不传入label_name参数，会自动使用PPG文件中的标签列名和标签值
    process_ppg_file(ppg_file, prv_file)
    
    # 或者批量处理整个文件夹
    # ppg_folder = "/home/czl/git_ppg/PFDM/dataset/PPG"
    # prv_folder = "/home/czl/git_ppg/PFDM/dataset/PRV"
    # batch_process_ppg_files(ppg_folder, prv_folder)
