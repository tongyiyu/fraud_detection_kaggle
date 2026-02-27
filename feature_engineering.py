# feature_engineering.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os

class FeatureEngineer:
    def __init__(self):
        self.numeric_features = []
        self.categorical_features = []
        self.preprocessor = None
        self.label_encoders = {}
        
    def identify_feature_types(self, df, target_col='fraud_reported'):
        """自动识别特征类型，并过滤全缺失的特征列"""
        # 1. 检查目标列是否存在（新增：防KeyError）
        if target_col not in df.columns:
            raise ValueError(f"目标列 '{target_col}' 不存在！数据集中的列名：{df.columns.tolist()}")
        
        # 2. 提取特征列（排除目标列）
        features = df.drop(columns=[target_col]).columns
        
        # 3. 区分数值/分类特征（原有逻辑）
        self.numeric_features = df[features].select_dtypes(
            include=['int64', 'float64']
        ).columns.tolist()
        
        self.categorical_features = df[features].select_dtypes(
            include=['object', 'category']
        ).columns.tolist()
        
        # ========== 新增核心逻辑：过滤全缺失列 ==========
        # 过滤数值特征：移除全缺失的列（至少有1个非缺失值才保留）
        self.numeric_features = [
            col for col in self.numeric_features 
            if df[col].notna().sum() > 0  # 统计非缺失值数量，大于0才保留
        ]
        
        # 过滤分类特征：同理移除全缺失列
        self.categorical_features = [
            col for col in self.categorical_features 
            if df[col].notna().sum() > 0
        ]
        
        # 4. 打印最终特征列表（原有逻辑，输出的是过滤后的结果）
        print(f"数值特征 ({len(self.numeric_features)}): {self.numeric_features}")
        print(f"分类特征 ({len(self.categorical_features)}): {self.categorical_features}")
        
        return self.numeric_features, self.categorical_features
    
    def handle_missing_values(self, df):
        """处理缺失值"""
        df_processed = df.copy()
        
        # 数值特征：中位数填充
        if self.numeric_features:
            numeric_imputer = SimpleImputer(strategy='median')
            df_processed[self.numeric_features] = numeric_imputer.fit_transform(
                df_processed[self.numeric_features]
            )
        
        # 分类特征：众数填充
        if self.categorical_features:
            for col in self.categorical_features:
                mode_val = df_processed[col].mode()
                if len(mode_val) > 0:
                    df_processed[col].fillna(mode_val[0])
                else:
                    df_processed[col].fillna('Unknown')
        
        return df_processed
    
    def encode_categorical_features(self, df, fit=True):
        """编码分类特征"""
        df_encoded = df.copy()
        
        if not self.categorical_features:
            return df_encoded
            
        if fit:
            # 初始化编码器
            self.label_encoders = {}
            
        for col in self.categorical_features:
            if fit:
                # 对于高基数分类变量使用LabelEncoder
                if df_encoded[col].nunique() > 10:
                    le = LabelEncoder()
                    # 处理NaN值
                    df_encoded[col] = df_encoded[col].astype(str)
                    le.fit(df_encoded[col])
                    self.label_encoders[col] = le
                else:
                    # 对于低基数使用OneHotEncoder（后续在Pipeline中处理）
                    continue
            
            # 应用编码
            if col in self.label_encoders:
                df_encoded[col] = self.label_encoders[col].transform(
                    df_encoded[col].astype(str)
                )
        
        return df_encoded
    
    def create_preprocessor(self):
        """创建预处理器Pipeline"""
        # 数值特征处理：缺失值 + 标准化
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # 分类特征处理：缺失值 + OneHot编码
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # 列转换器
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features),
                ('cat', categorical_transformer, self.categorical_features)
            ],
            remainder='passthrough'
        )
        
        return self.preprocessor
    
    def fit_transform(self, df, target_col='fraud_reported'):
        """完整的特征工程流程"""
        print("开始特征工程...")
        
        # 1. 识别特征类型
        self.identify_feature_types(df, target_col)
        
        # 2. 处理缺失值
        df_processed = self.handle_missing_values(df)
        print("✅ 缺失值处理完成")
        
        # 3. 编码分类特征（LabelEncoder部分）
        df_encoded = self.encode_categorical_features(df_processed, fit=True)
        print("✅ 分类特征编码完成")
        
        # 4. 准备目标变量
        X = df_encoded.drop(columns=[target_col])
        y = df_encoded[target_col]
        
        # 5. 创建并应用预处理器（OneHot + 标准化）
        preprocessor = self.create_preprocessor()
        X_processed = preprocessor.fit_transform(X)
        
        # 获取特征名称
        feature_names = self._get_feature_names(preprocessor)
        
        print(f"✅ 特征工程完成！最终特征数量: {X_processed.shape[1]}")
        
        return X_processed, y, feature_names, preprocessor
    
    def _get_feature_names(self, preprocessor):
        """获取处理后的特征名称"""
        feature_names = []
        
        # 数值特征名称
        if self.numeric_features:
            feature_names.extend(self.numeric_features)
        
        # 分类特征名称（OneHot编码后）
        if self.categorical_features:
            cat_encoder = preprocessor.named_transformers_['cat']
            onehot_encoder = cat_encoder.named_steps['onehot']
            cat_feature_names = onehot_encoder.get_feature_names_out(self.categorical_features)
            feature_names.extend(cat_feature_names)
        
        return feature_names
    
    def save_pipeline(self, filepath):
        """保存预处理器和编码器（自动创建目标目录）"""
        # ========== 新增核心逻辑：创建保存目录（如果不存在） ==========
        # 提取文件路径中的目录部分（比如从'models/feature_pipeline.pkl'中提取'models/'）
        save_dir = os.path.dirname(filepath)
        # 如果目录不为空且不存在，则递归创建（支持多级目录，如'a/b/c/'）
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)  # exist_ok=True避免目录已存在时报错
            print(f"✅ 自动创建保存目录：{save_dir}")
        
        # ========== 原有保存逻辑（完全保留） ==========
        pipeline_data = {
            'preprocessor': self.preprocessor,
            'label_encoders': self.label_encoders,
            'numeric_features': self.numeric_features,
            'categorical_features': self.categorical_features
        }
        joblib.dump(pipeline_data, filepath)
        print(f"✅ 特征工程Pipeline已保存: {filepath}")

# 使用示例
if __name__ == "__main__":
    # 读取数据
    df = pd.read_csv('data/train.csv')
    
    # 创建特征工程师
    fe = FeatureEngineer()
    
    # 执行特征工程
    X, y, feature_names, preprocessor = fe.fit_transform(df)
    
    # 保存处理后的数据
    os.makedirs('processed_data', exist_ok=True)
    np.save('processed_data/X_train.npy', X)
    np.save('processed_data/y_train.npy', y)
    
    # 保存特征名称
    with open('processed_data/feature_names.txt', 'w') as f:
        f.write('\n'.join(feature_names))
    
    # 保存Pipeline
    fe.save_pipeline('models/feature_pipeline.pkl')
    
    print(f"处理后数据形状: {X.shape}")
    print(f"特征名称示例: {feature_names[:5]}")