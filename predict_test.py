# predict_test.py
import pandas as pd
import numpy as np
import joblib
import os

def predict_test_set():
    """对测试集进行预测"""
    print("开始测试集预测...")
    
    # 加载测试数据
    test_df = pd.read_csv('data/test.csv')
    print(f"测试集形状: {test_df.shape}")
    
    # 加载特征工程Pipeline
    pipeline_data = joblib.load('models/feature_pipeline.pkl')
    preprocessor = pipeline_data['preprocessor']
    label_encoders = pipeline_data['label_encoders']
    numeric_features = pipeline_data['numeric_features']
    categorical_features = pipeline_data['categorical_features']
    
    # 应用相同的预处理步骤
    test_processed = test_df.copy()
    
    # 处理缺失值（使用训练时的策略）
    if numeric_features:
        # 数值特征：使用中位数（从preprocessor中获取）
        numeric_imputer = preprocessor.named_transformers_['num'].named_steps['imputer']
        test_processed[numeric_features] = numeric_imputer.transform(test_processed[numeric_features])
    
    if categorical_features:
        # 分类特征：先用LabelEncoder处理高基数变量
        for col in categorical_features:
            if col in label_encoders:
                # 处理新类别（训练时未见过的）
                test_processed[col] = test_processed[col].astype(str)
                # 将未知类别映射为特殊值
                known_classes = set(label_encoders[col].classes_)
                test_processed[col] = test_processed[col].apply(
                    lambda x: x if x in known_classes else 'Unknown'
                )
                # 更新编码器以包含'Unknown'
                if 'Unknown' not in label_encoders[col].classes_:
                    label_encoders[col].classes_ = np.append(label_encoders[col].classes_, 'Unknown')
                test_processed[col] = label_encoders[col].transform(test_processed[col])
    
    # 应用完整的预处理器（OneHot + 标准化）
    X_test = test_processed.drop(columns=['id'], errors='ignore')  # 假设id是标识列
    X_test_processed = preprocessor.transform(X_test)
    
    print(f"处理后测试集形状: {X_test_processed.shape}")
    
    # 加载最佳模型
    best_model = joblib.load('models/tuned_xgboost_model.pkl')
    
    # 进行预测
    y_pred_proba = best_model.predict_proba(X_test_processed)[:, 1]
    
    # 创建提交文件
    submission_df = pd.DataFrame({
        'id': test_df['id'],
        'fraud_probability': y_pred_proba
    })
    
    # 保存预测结果
    os.makedirs('submissions', exist_ok=True)
    submission_df.to_csv('submissions/submission_v1.csv', index=False)
    
    print("✅ 测试集预测完成！")
    print(f"预测概率范围: [{y_pred_proba.min():.4f}, {y_pred_proba.max():.4f}]")
    print(f"平均预测概率: {y_pred_proba.mean():.4f}")
    
    return submission_df

if __name__ == "__main__":
    submission = predict_test_set()