"""
从PPG信号中提取PRV（脉率变异性）数据

修正后的PRV提取算法：
| 步骤 | 说明 |
|------|------|
| 1. 滤波 | 带通滤波 0.5-8Hz（适合PPG信号的心率频率范围）|
| 2. 峰值检测 | 基于采样率动态计算distance参数 |
| 3. PP间期计算 | 转换为毫秒(ms)单位，具有生理意义 |
| 4. 异常值处理 | 基于生理合理心率范围（40-200bpm）|
| 5. 归一化 | 缩放到0-1范围，保持相对变异特征 |
| 6. 长度调整 | 使用线性插值调整到固定长度80 |
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks
from itertools import chain
import os


# 默认参数
DEFAULT_RECORDING_TIME = 90  # 默认记录时长（秒）
TARGET_PRV_LENGTH = 80  # 目标PRV序列长度

# 生理合理的心率范围
MIN_HR_BPM = 40  # 最低心率（bpm）
MAX_HR_BPM = 200  # 最高心率（bpm）


def sampling_freq(signal, time=DEFAULT_RECORDING_TIME):
    """计算采样频率"""
    return len(signal) / time


def butter_bandpass(lowcut, highcut, fs, order=4):
    """
    巴特沃斯带通滤波器
    
    参数:
        lowcut: 低截止频率(Hz)
        highcut: 高截止频率(Hz)
        fs: 采样频率(Hz)
        order: 滤波器阶数
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    # 确保high不超过Nyquist频率的99%
    high = min(highcut / nyq, 0.99)
    b, a = butter(order, [low, high], btype='band')
    return b, a


def apply_filtering(signal, fs=None):
    """
    应用带通滤波
    
    参数:
        signal: PPG信号
        fs: 采样频率，如果为None则自动计算
    
    说明:
        使用0.5-8Hz带通滤波，这个范围覆盖了正常心率（30-240bpm对应0.5-4Hz）
        同时保留了PPG波形的主要特征
    """
    if fs is None:
        fs = sampling_freq(signal)
    print(f"采样频率: {fs:.1f} Hz")
    
    # PPG信号的有效频率范围通常是0.5-8Hz
    # 0.5Hz对应30bpm，8Hz可以捕获PPG波形的高频成分
    b, a = butter_bandpass(0.5, 8, fs, order=4)
    return filtfilt(b, a, signal)


def systolic_peaks_1(signal, fs=None):
    """
    检测收缩峰值
    
    参数:
        signal: 滤波后的PPG信号
        fs: 采样频率
    
    说明:
        distance参数基于最大心率(200bpm)计算
        200bpm = 3.33Hz，即每个心跳间隔至少0.3秒
        distance = 0.3 * fs
    """
    if fs is None:
        fs = sampling_freq(signal)
    
    # 基于最大心率计算最小峰间距
    # 200bpm对应0.3秒的最小间隔
    min_distance = int(0.3 * fs)
    min_distance = max(min_distance, 5)  # 确保至少5个样本
    
    # 使用prominence参数提高峰值检测的可靠性
    peaks, properties = find_peaks(signal, distance=min_distance, prominence=0.1*np.std(signal))
    
    return peaks


def tfn_points_1(signal, fs=None):
    """
    检测谷值点
    
    参数:
        signal: 滤波后的PPG信号
        fs: 采样频率
    """
    if fs is None:
        fs = sampling_freq(signal)
    
    min_distance = int(0.3 * fs)
    min_distance = max(min_distance, 5)
    
    return find_peaks(signal * (-1), height=0, distance=min_distance)[0]


def extract_PRV(PPG, recording_time=DEFAULT_RECORDING_TIME):
    """
    提取脉率变异性
    
    参数:
        PPG: PPG信号数组
        recording_time: 记录时长（秒），默认90秒
    
    返回:
        PRV: 长度为80的PRV数组（归一化后的PP间期序列）
    """
    # 计算采样频率
    fs = sampling_freq(PPG, recording_time)
    
    # 1. 带通滤波
    fil = apply_filtering(PPG, fs)
    
    # 2. 峰值检测
    systolics = systolic_peaks_1(fil, fs)
    
    # 检查是否检测到足够的峰值
    if len(systolics) < 3:
        print(f"警告: 仅检测到 {len(systolics)} 个峰值，数据可能有问题")
        # 返回默认值
        return np.ones(TARGET_PRV_LENGTH) * 0.5
    
    # 3. 计算PP间期（转换为毫秒）
    # PP间期 = (峰值位置差) / 采样率 * 1000ms
    pp_intervals_samples = np.diff(systolics)
    pp_intervals_ms = pp_intervals_samples / fs * 1000  # 转换为毫秒
    
    # 4. 异常值处理 - 基于生理合理范围
    # 40-200bpm 对应 300-1500ms 的PP间期
    min_pp_ms = 60000 / MAX_HR_BPM  # 300ms (200bpm)
    max_pp_ms = 60000 / MIN_HR_BPM  # 1500ms (40bpm)
    
    pp_intervals_cleaned = remove_outliers(pp_intervals_ms, min_pp_ms, max_pp_ms)
    
    # 5. 归一化到0-1范围
    # 使用min-max归一化，保持相对变异特征
    pp_min = min_pp_ms
    pp_max = max_pp_ms
    pp_normalized = (pp_intervals_cleaned - pp_min) / (pp_max - pp_min)
    pp_normalized = np.clip(pp_normalized, 0, 1)  # 确保在0-1范围内
    
    # 6. 调整长度到TARGET_PRV_LENGTH（使用插值而非简单截取/填充）
    prv_resampled = resample_to_length(pp_normalized, TARGET_PRV_LENGTH)
    
    return prv_resampled


def remove_outliers(pp_intervals, min_val, max_val):
    """
    移除PP间期中的异常值
    
    参数:
        pp_intervals: PP间期数组（毫秒）
        min_val: 最小合理值
        max_val: 最大合理值
    
    返回:
        处理后的PP间期数组
    """
    pp_cleaned = pp_intervals.copy()
    
    # 使用中位数作为参考值
    median_val = np.median(pp_cleaned)
    
    for i in range(len(pp_cleaned)):
        if pp_cleaned[i] < min_val or pp_cleaned[i] > max_val:
            # 异常值替换策略
            if i == 0:
                # 第一个值异常，使用后一个值或中位数
                pp_cleaned[i] = pp_cleaned[i+1] if len(pp_cleaned) > 1 and min_val <= pp_cleaned[i+1] <= max_val else median_val
            elif i == len(pp_cleaned) - 1:
                # 最后一个值异常，使用前一个值
                pp_cleaned[i] = pp_cleaned[i-1]
            else:
                # 中间值异常，使用相邻值的平均
                pp_cleaned[i] = (pp_cleaned[i-1] + pp_cleaned[i+1]) / 2
    
    # 二次检查，确保所有值都在合理范围内
    pp_cleaned = np.clip(pp_cleaned, min_val, max_val)
    
    return pp_cleaned


def resample_to_length(data, target_length):
    """
    使用线性插值将数据重采样到目标长度
    
    参数:
        data: 原始数据数组
        target_length: 目标长度
    
    返回:
        重采样后的数组
    """
    if len(data) == 0:
        return np.ones(target_length) * 0.5
    
    if len(data) == target_length:
        return data
    
    # 使用线性插值
    original_indices = np.linspace(0, 1, len(data))
    target_indices = np.linspace(0, 1, target_length)
    resampled = np.interp(target_indices, original_indices, data)
    
    return resampled


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
    
    # 保存PRV数据，包含表头，保留9位小数
    prv_df.to_csv(output_file_path, index=False, header=True, float_format='%.9f')
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
    # # 示例：处理单个文件
    # ppg_file = "/home/czl/git_ppg/PFDM/dataset/PPG/Anxiety.csv"
    # prv_file = "/home/czl/git_ppg/PFDM/dataset/PRV/Anxiety_extracted.csv"
    
    # # 不传入label_name参数，会自动使用PPG文件中的标签列名和标签值
    # process_ppg_file(ppg_file, prv_file)
    
    # 或者批量处理整个文件夹
    ppg_folder = "/home/czl/git_ppg/PFDM/dataset/PPG"
    prv_folder = "/home/czl/git_ppg/PFDM/dataset/newPRV"
    batch_process_ppg_files(ppg_folder, prv_folder)
