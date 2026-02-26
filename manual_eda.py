# manual_eda.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 设置中文字体（Windows）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
df = pd.read_csv('data/train.csv')
print(f"数据形状: {df.shape}")
print(f"列名: {list(df.columns)}")

# 查看基本信息
print("\n=== 数据基本信息 ===")
print(df.info())

print("\n=== 缺失值统计 ===")
missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
print(missing)

print("\n=== 目标变量分布 ===")
target_dist = df['fraud'].value_counts()
print(target_dist)
print(f"欺诈比例: {target_dist[1]/len(df)*100:.2f}%")

# 创建可视化目录
os.makedirs('visualizations', exist_ok=True)

# 目标变量分布图
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='fraud')
plt.title('欺诈标签分布')
plt.savefig('visualizations/target_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# 数值特征分布（选择几个重要特征）
numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
if 'fraud' in numeric_cols:
    numeric_cols.remove('fraud')

# 选择前6个数值特征
selected_numeric = numeric_cols[:6]
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, col in enumerate(selected_numeric):
    sns.histplot(data=df, x=col, hue='fraud', ax=axes[i], alpha=0.7)
    axes[i].set_title(f'{col} 分布')

plt.tight_layout()
plt.savefig('visualizations/numeric_features.png', dpi=300, bbox_inches='tight')
plt.close()

# 相关性热图
plt.figure(figsize=(12, 10))
correlation_matrix = df[selected_numeric + ['fraud']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('特征相关性热图')
plt.savefig('visualizations/correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ 手动EDA完成，图表保存在 visualizations/ 目录")