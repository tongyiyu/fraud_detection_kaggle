# app.py
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)

# 全局变量：加载模型和预处理器
model = None
preprocessor = None
label_encoders = None
feature_names = None

def load_model():
    """加载模型和预处理器"""
    global model, preprocessor, label_encoders, feature_names
    
    # 加载特征工程Pipeline
    pipeline_data = joblib.load('models/feature_pipeline.pkl')
    preprocessor = pipeline_data['preprocessor']
    label_encoders = pipeline_data['label_encoders']
    
    # 加载模型
    model = joblib.load('models/tuned_xgboost_model.pkl')
    
    # 加载特征名称
    with open('processed_data/feature_names.txt', 'r') as f:
        feature_names = [line.strip() for line in f.readlines()]
    
    print("✅ 模型和预处理器加载成功")

@app.route('/predict', methods=['POST'])
def predict():
    """预测API端点"""
    try:
        # 获取JSON数据
        data = request.get_json()
        
        # 转换为DataFrame
        df = pd.DataFrame([data]) if isinstance(data, dict) else pd.DataFrame(data)
        
        # 应用预处理（简化版本，实际需要更完整的处理）
        # 这里假设输入数据已经过适当预处理
        
        # 进行预测
        predictions = model.predict_proba(preprocessor.transform(df))[:, 1]
        
        # 返回结果
        results = []
        for i, pred in enumerate(predictions):
            results.append({
                'id': df.iloc[i].get('id', i),
                'fraud_probability': float(pred),
                'prediction': int(pred > 0.5)
            })
        
        return jsonify({
            'status': 'success',
            'predictions': results
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查端点"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    load_model()
    app.run(host='0.0.0.0', port=5000, debug=False)