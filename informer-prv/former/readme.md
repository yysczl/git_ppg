# 项目文件结构
informer-prv/
├── data_processor.py          # 原有文件
├── models.py                  # 原有文件
├── informer_model.py          # 原有文件
├── model_trainer.py           # 原有文件
├── ppg_informer.py            # 原有文件
├── prv_informer.py            # 原有文件
├── plot_utils.py              # 原有文件
├── evaluate_model.py          # 原有文件
├── requirements.txt           # 原有文件
├── README.md                  # 原有文件
│
├── ppg_former_model.py        # 【新建】PPG-Former模块
├── dual_stream_model.py       # 【新建】双流网络+完整模型
├── multi_task_trainer.py      # 【新建】多任务训练器
└── ppg_former_main.py         # 【新建】主程序入口

# 使用方法

# 进入项目目录
cd informer-prv
# 运行新模型
python ppg_former_main.py

