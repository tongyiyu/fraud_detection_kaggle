# verify_features.py
import numpy as np

# 加载处理后的数据
X = np.load('processed_data/X_train.npy')
y = np.load('processed_data/y_train.npy')

print(f"X形状: {X.shape}")
print(f"y形状: {y.shape}")
print(f"X数据类型: {X.dtype}")
print(f"X前5行:\n{X[:5]}")
print(f"y分布: {np.bincount(y)}")