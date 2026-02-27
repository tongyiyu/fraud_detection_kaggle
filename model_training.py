# model_training.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

class ModelTrainer:
    def __init__(self):
        self.models = {
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'xgboost': XGBClassifier(random_state=42, eval_metric='logloss')
        }
        self.results = {}
        self.best_model = None
        self.best_score = 0
        
    def train_and_evaluate(self, X, y, cv_folds=5):
        """训练和评估所有模型"""
        print("开始模型训练和评估...")
        
        # 创建结果目录
        os.makedirs('results', exist_ok=True)
        
        # 分层K折交叉验证（处理不平衡数据）
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        for name, model in self.models.items():
            print(f"\n训练 {name}...")
            
            # 交叉验证
            cv_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
            
            # 训练完整模型
            model.fit(X, y)
            
            # 预测
            y_pred = model.predict(X)
            y_pred_proba = model.predict_proba(X)[:, 1]
            
            # 计算指标
            auc_score = roc_auc_score(y, y_pred_proba)
            accuracy = np.mean(y_pred == y)
            
            # 保存结果
            self.results[name] = {
                'model': model,
                'cv_scores': cv_scores,
                'mean_cv_score': cv_scores.mean(),
                'std_cv_score': cv_scores.std(),
                'auc_score': auc_score,
                'accuracy': accuracy,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            print(f"{name} - CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
            print(f"{name} - Full AUC: {auc_score:.4f}")
            
            # 保存分类报告
            report = classification_report(y, y_pred, output_dict=True)
            self.results[name]['classification_report'] = report
            
            # 保存混淆矩阵
            cm = confusion_matrix(y, y_pred)
            self.results[name]['confusion_matrix'] = cm
        
        # 选择最佳模型
        best_model_name = max(self.results.keys(), key=lambda k: self.results[k]['mean_cv_score'])
        self.best_model = self.results[best_model_name]['model']
        self.best_score = self.results[best_model_name]['mean_cv_score']
        
        print(f"\n✅ 最佳模型: {best_model_name} (CV AUC: {self.best_score:.4f})")
        
        return self.results
    
    def plot_results(self):
        """绘制结果可视化"""
        # 创建可视化目录
        os.makedirs('visualizations/models', exist_ok=True)
        
        # 模型性能比较
        model_names = list(self.results.keys())
        cv_scores = [self.results[name]['mean_cv_score'] for name in model_names]
        std_scores = [self.results[name]['std_cv_score'] for name in model_names]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(model_names, cv_scores, yerr=std_scores, capsize=5, alpha=0.7)
        plt.ylabel('Cross-Validation AUC Score')
        plt.title('Model Performance Comparison')
        plt.ylim(0.7, 1.0)
        
        # 添加数值标签
        for bar, score in zip(bars, cv_scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{score:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('visualizations/models/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 混淆矩阵可视化
        for name, result in self.results.items():
            plt.figure(figsize=(6, 5))
            sns.heatmap(result['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
            plt.title(f'{name} - Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.savefig(f'visualizations/models/{name}_confusion_matrix.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        print("✅ 模型结果可视化已保存")
    
    def save_best_model(self, filepath):
        """保存最佳模型"""
        joblib.dump(self.best_model, filepath)
        print(f"✅ 最佳模型已保存: {filepath}")
    
    def get_feature_importance(self, feature_names):
        """获取特征重要性"""
        importance_results = {}
        
        for name, result in self.results.items():
            model = result['model']
            importance = None
            
            if hasattr(model, 'feature_importances_'):
                # 树模型
                importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                # 线性模型
                importance = np.abs(model.coef_[0])
            
            if importance is not None:
                # 创建特征重要性DataFrame
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importance
                }).sort_values('importance', ascending=False)
                
                importance_results[name] = importance_df
                
                # 保存前20个重要特征
                top_features = importance_df.head(20)
                top_features.to_csv(f'results/{name}_feature_importance.csv', index=False)
                
                # 绘制特征重要性图
                plt.figure(figsize=(10, 8))
                sns.barplot(data=top_features, y='feature', x='importance')
                plt.title(f'{name} - Top 20 Feature Importance')
                plt.tight_layout()
                plt.savefig(f'visualizations/models/{name}_feature_importance.png', dpi=300, bbox_inches='tight')
                plt.close()
        
        return importance_results

if __name__ == "__main__":
    # 加载处理后的数据
    X = np.load('processed_data/X_train.npy')
    y = np.load('processed_data/y_train.npy')
    
    # 检查缺失值
    print("=== 数据缺失值检查 ===")
    print(f"X的形状: {X.shape}")
    print(f"X中NaN的数量: {np.isnan(X).sum()}")
    print(f"X中NaN的比例: {np.isnan(X).sum() / X.size:.6f}")
    print(f"是否有无限值: {np.isinf(X).sum()}")

    # 若有NaN，定位并处理全缺失列
    nan_total = np.isnan(X).sum()
    if nan_total > 0:
        nan_rows = np.any(np.isnan(X), axis=1)
        nan_cols = np.any(np.isnan(X), axis=0)
        print(f"包含NaN的行数: {nan_rows.sum()}")
        print(f"包含NaN的列数: {nan_cols.sum()}")
        
        # 定位NaN列的索引
        nan_col_indices = np.where(nan_cols)[0]
        print(f"NaN列的索引: {nan_col_indices}")
        
        # 加载特征名称（提前加载用于定位NaN列名称）
        with open('processed_data/feature_names.txt', 'r') as f:
            feature_names = [line.strip() for line in f.readlines()]
        
        # 打印NaN列对应的特征名称（若特征名称数量匹配）
        if len(nan_col_indices) > 0 and len(feature_names) == X.shape[1]:
            nan_col_names = [feature_names[idx] for idx in nan_col_indices]
            print(f"NaN列对应的特征名称: {nan_col_names}")
        
        # ========== 核心修复：删除全缺失列 ==========
        X = X[:, ~nan_cols]  # 删除所有包含NaN的列（此处仅1列）
        # 同步更新特征名称（删除对应列的名称）
        feature_names = [name for idx, name in enumerate(feature_names) if idx not in nan_col_indices]
        print(f"\n✅ 已删除全缺失列，处理后X形状: {X.shape}")
        print(f"✅ 处理后NaN数量: {np.isnan(X).sum()}")
    else:
        # 无NaN时正常加载特征名称
        with open('processed_data/feature_names.txt', 'r') as f:
            feature_names = [line.strip() for line in f.readlines()]

    # 创建模型训练器
    trainer = ModelTrainer()

    # 训练和评估模型
    results = trainer.train_and_evaluate(X, y)

    # 绘制结果
    trainer.plot_results()

    # 获取特征重要性（传入更新后的特征名称）
    importance_results = trainer.get_feature_importance(feature_names)

    # 保存最佳模型
    trainer.save_best_model('models/best_model.pkl')

    # 打印详细结果
    for name, result in results.items():
        print(f"\n=== {name.upper()} ===")
        print(f"CV AUC: {result['mean_cv_score']:.4f} ± {result['std_cv_score']:.4f}")
        print(f"Full AUC: {result['auc_score']:.4f}")
        print("Classification Report:")
        print(pd.DataFrame(result['classification_report']).T)
