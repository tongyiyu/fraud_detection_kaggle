# hyperparameter_tuning.py
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import make_scorer, roc_auc_score
from xgboost import XGBClassifier
from scipy.stats import uniform, randint
import joblib
import os
import time

class HyperparameterTuner:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.best_params = None
        self.best_score = 0
        self.tuning_results = {}
        
        # 计算正负样本比例（用于scale_pos_weight）
        neg_count = np.sum(y == 0)
        pos_count = np.sum(y == 1)
        self.scale_pos_weight = neg_count / pos_count
        
    def grid_search_tuning(self):
        """网格搜索调优"""
        print("开始网格搜索调优...")
        
        # 参数网格
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
            'scale_pos_weight': [self.scale_pos_weight]
        }
        
        # 创建模型
        xgb = XGBClassifier(random_state=42, eval_metric='logloss')
        
        # 网格搜索
        grid_search = GridSearchCV(
            estimator=xgb,
            param_grid=param_grid,
            scoring='roc_auc',
            cv=3,  # 减少CV折数以节省时间
            n_jobs=-1,
            verbose=1
        )
        
        start_time = time.time()
        grid_search.fit(self.X, self.y)
        end_time = time.time()
        
        # 保存结果
        self.tuning_results['grid_search'] = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'time_taken': end_time - start_time,
            'cv_results': grid_search.cv_results_
        }
        
        print(f"✅ 网格搜索完成！最佳分数: {grid_search.best_score_:.4f}")
        print(f"最佳参数: {grid_search.best_params_}")
        print(f"耗时: {end_time - start_time:.2f} 秒")
        
        return grid_search.best_estimator_
    
    def bayesian_optimization_tuning(self):
        """贝叶斯优化调优（使用Optuna）"""
        try:
            import optuna
            
            print("开始贝叶斯优化调优...")
            
            def objective(trial):
                # 定义参数空间
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'scale_pos_weight': self.scale_pos_weight,
                    'random_state': 42,
                    'eval_metric': 'logloss'
                }
                
                # 创建模型
                model = XGBClassifier(**params)
                
                # 交叉验证评分
                cv_scores = cross_val_score(model, self.X, self.y, cv=3, scoring='roc_auc')
                return cv_scores.mean()
            
            # 创建研究对象
            study = optuna.create_study(direction='maximize')
            
            start_time = time.time()
            study.optimize(objective, n_trials=50, show_progress_bar=True)
            end_time = time.time()
            
            # 保存结果
            self.tuning_results['bayesian_optimization'] = {
                'best_params': study.best_params,
                'best_score': study.best_value,
                'time_taken': end_time - start_time,
                'study': study
            }
            
            print(f"✅ 贝叶斯优化完成！最佳分数: {study.best_value:.4f}")
            print(f"最佳参数: {study.best_params}")
            print(f"耗时: {end_time - start_time:.2f} 秒")
            
            # 训练最佳模型
            best_params = study.best_params.copy()
            best_params.update({
                'scale_pos_weight': self.scale_pos_weight,
                'random_state': 42,
                'eval_metric': 'logloss'
            })
            best_model = XGBClassifier(**best_params)
            best_model.fit(self.X, self.y)
            
            return best_model
            
        except ImportError:
            print("⚠️ Optuna未安装，跳过贝叶斯优化")
            print("安装命令: pip install optuna")
            return None
    
    def compare_tuning_methods(self):
        """比较不同调优方法的结果"""
        print("\n=== 超参数调优方法比较 ===")
        
        methods = []
        scores = []
        times = []
        
        if 'grid_search' in self.tuning_results:
            methods.append('Grid Search')
            scores.append(self.tuning_results['grid_search']['best_score'])
            times.append(self.tuning_results['grid_search']['time_taken'])
            
        if 'bayesian_optimization' in self.tuning_results:
            methods.append('Bayesian Optimization')
            scores.append(self.tuning_results['bayesian_optimization']['best_score'])
            times.append(self.tuning_results['bayesian_optimization']['time_taken'])
        
        # 创建比较DataFrame
        comparison_df = pd.DataFrame({
            'Method': methods,
            'Best Score': scores,
            'Time (seconds)': times
        })
        
        print(comparison_df.to_string(index=False))
        
        # 选择最佳方法
        best_method_idx = np.argmax(scores)
        best_method = methods[best_method_idx]
        self.best_params = (self.tuning_results['grid_search']['best_params'] 
                          if best_method == 'Grid Search' 
                          else self.tuning_results['bayesian_optimization']['best_params'])
        self.best_score = scores[best_method_idx]
        
        print(f"\n🏆 最佳调优方法: {best_method}")
        print(f"最佳参数: {self.best_params}")
        
        return comparison_df
    
    def save_tuning_results(self):
        """保存调优结果"""
        os.makedirs('results/tuning', exist_ok=True)
        
        # 保存最佳参数
        with open('results/tuning/best_params.txt', 'w') as f:
            f.write(str(self.best_params))
        
        # 保存详细结果
        joblib.dump(self.tuning_results, 'results/tuning/tuning_results.pkl')
        
        print("✅ 调优结果已保存")

# 使用示例
if __name__ == "__main__":
    # 安装optuna（如果需要）
    # pip install optuna
    
    # 加载数据
    X = np.load('processed_data/X_train.npy')
    y = np.load('processed_data/y_train.npy')
    
    # 创建调优器
    tuner = HyperparameterTuner(X, y)
    
    # 执行网格搜索
    grid_model = tuner.grid_search_tuning()
    
    # 执行贝叶斯优化（可选）
    bayesian_model = tuner.bayesian_optimization_tuning()
    
    # 比较结果
    comparison_df = tuner.compare_tuning_methods()
    
    # 保存结果
    tuner.save_tuning_results()
    
    # 保存最佳模型（基于网格搜索结果）
    if grid_model is not None:
        joblib.dump(grid_model, 'models/tuned_xgboost_model.pkl')
        print("✅ 调优后的模型已保存: models/tuned_xgboost_model.pkl")