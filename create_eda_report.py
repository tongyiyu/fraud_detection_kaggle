# create_eda_report.py
import pandas as pd
# from pandas_profiling import ProfileReport
from ydata_profiling import ProfileReport
import os

# 创建输出目录
os.makedirs('reports', exist_ok=True)

# 读取训练数据
df = pd.read_csv('data/train.csv')

# 生成EDA报告
profile = ProfileReport(df, title="保险反欺诈数据集 EDA报告", explorative=True)
profile.to_file("reports/eda_report.html")

print("✅ EDA报告已生成: reports/eda_report.html")