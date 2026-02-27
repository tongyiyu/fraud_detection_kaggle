
### 📝 5. 记录学习笔记（20分钟）
创建 `2024-02-15-hyperparameter_tuning_notes.md`：
```markdown
# 超参数调优学习笔记 - 2024-02-15

## ✅ 完成内容
- 实现网格搜索超参数调优
- 实现贝叶斯优化（Optuna）超参数调优
- 比较不同调优方法的效果和效率
- 应用最佳参数训练最终模型
- 保存调优结果和最佳模型

## 💡 关键收获
### 超参数调优最佳实践
1. **参数范围设定**：
   - 基于领域知识和初步实验
   - 避免过宽或过窄的搜索空间
   - 对数尺度搜索学习率等参数

2. **交叉验证设置**：
   - 使用分层K折（StratifiedKFold）处理不平衡数据
   - 减少CV折数（如3折）以节省调优时间
   - 使用相同的随机种子确保可复现性

3. **评估指标选择**：
   - 对于不平衡数据，优先选择AUC-ROC而非准确率
   - 考虑业务需求（如更重视召回率还是精确率）

### 贝叶斯优化优势
- **智能搜索**：基于历史试验结果指导下一步搜索
- **效率更高**：通常比网格搜索和随机搜索更快找到最优解
- **自适应**：自动在 promising 区域进行更密集的搜索

## ❌ 遇到的问题与解决
| 问题 | 解决方案 |
|------|----------|
| 调优时间过长 | 减少CV折数、缩小参数范围、使用更少的试验次数 |
| 内存不足 | 限制并行进程（n_jobs=1）、分批处理 |
| Optuna安装失败 | 使用conda安装：`conda install -c conda-forge optuna` |

## 📌 代码模板
```python
# 贝叶斯优化模板（Optuna）
import optuna

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
    }
    
    model = XGBClassifier(**params)
    cv_scores = cross_val_score(model, X, y, cv=3, scoring='roc_auc')
    return cv_scores.mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)