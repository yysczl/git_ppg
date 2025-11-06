# 脉搏波时序数据分析与预测

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.7+](https://img.shields.io/badge/pytorch-1.7+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

本项目使用多种深度学习模型对人体脉搏波时序数据进行分析和预测，数据来源于试验者在观看不同情绪诱发视频时的脉搏波记录。

## 项目结构

- `ppg_informer.py`: PPG数据主程序，包含数据预处理、模型训练和评估流程
- `prv_informer.py`: PRV数据主程序，包含数据预处理、模型训练和评估流程
- `informer_model.py`: Informer模型架构实现
- `models.py`: 多个时序模型架构实现（包括LSTM、GRU、BiLSTM、TCN、Transformer等）
- `evaluate_model.py`: 加载模型并评估
- `plot_utils.py`: 画图工具函数
- `data_processor.py`: 预处理ppg和prv信号的函数
- `model_trainer.py`: 训练模型
- `requirements.txt`: 项目依赖包列表

## 数据说明

### PPG数据
PPG数据文件位于 `dataset/PPG/` 目录下，包含90名试验者的数据：
- 每行代表一个试验者
- 前1800列为脉搏波数据点（时序特征）
- 最后一列为量表压力数值（标签值）

### PRV数据
PRV数据文件位于 `dataset/PRV/` 目录下，包含90名试验者的数据：
- 每行代表一个试验者
- 前80列为脉搏波数据点（时序特征）
- 最后一列为量表压力数值（标签值）

## 支持的情绪类型

- Anxiety（焦虑）
- Happy（快乐）
- Peace（平静）
- Sad（悲伤）
- Stress（压力）

## 功能实现

1. **数据预处理**
   - 读取CSV文件，提取特征和标签
   - 按8:1:1比例划分训练集、验证集和测试集
   - 对数据进行标准化处理
   - 支持K折交叉验证

2. **模型架构**
   - 支持多种时序模型：Informer、LSTM、GRU、BiLSTM、TCN、Transformer、AttentionRNN、CNN1D、MLP、Seq2Seq、NBeats
   - 实现多头注意力机制
   - 可配置的模型参数
   - 配置适当的输入/输出维度

3. **训练与评估**
   - 使用MSE作为损失函数
   - 计算MAE和RMSE作为评估指标
   - 在验证集上监控模型表现
   - 在测试集上评估最终性能
   - 支持学习率调度
   - 支持早停机制

4. **可视化**
   - 绘制训练过程中损失、MAE和RMSE的变化曲线
   - 绘制测试集上预测结果与真实标签的对比散点图
   - 可视化模型预测效果

## 依赖项

- Python 3.7+
- PyTorch 1.7+
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- 其他依赖项请参考 [requirements.txt](requirements.txt)

## 使用方法

1. 安装依赖包
```
pip install -r requirements.txt
```

2. 选择模型类型

在 `ppg_informer.py` 和 `prv_informer.py` 文件中，通过修改 `model_type` 变量选择不同的模型：
- `Informer`: Informer模型
- `LSTM`: 长短期记忆网络
- `GRU`: 门控循环单元
- `BiLSTM`: 双向LSTM
- `TCN`: 时间卷积网络
- `Transformer`: Transformer模型
- `AttentionRNN`: 基于注意力机制的RNN模型
- `CNN1D`: 一维卷积神经网络
- `MLP`: 多层感知机
- `Seq2Seq`: 序列到序列模型
- `NBeats`: N-BEATS时间序列预测模型

3. 运行PPG数据主程序
```
python ppg_informer.py
```

4. 运行PRV数据主程序
```
python prv_informer.py
```

5. 评估已保存的模型
```
python evaluate_model.py
```

## 结果输出

- `training_process.png`: 训练过程中损失和评估指标的变化曲线
- `predictions.png`: 测试集上预测结果与真实标签的对比散点图
- `final_[model_type]_model.pth`: 训练好的模型文件
- 控制台输出：训练过程中的损失和评估指标，以及测试集的最终评估结果