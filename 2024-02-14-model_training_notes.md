# 模型训练学习笔记 - 2024-02-14

## ✅ 完成内容
- 实现三种机器学习模型（逻辑回归、随机森林、XGBoost）
- 使用分层K折交叉验证处理不平衡数据
- 多维度评估指标（AUC、准确率、分类报告、混淆矩阵）
- 特征重要性分析
- 结果可视化
- 最佳模型持久化

## 💡 关键收获
### 不平衡数据处理策略
1. **分层交叉验证**：确保每个fold中正负样本比例一致
   ```python
   from sklearn.model_selection import StratifiedKFold
   cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
2. 评估指标选择：
 - AUC-ROC：对类别不平衡相对鲁棒
 - Precision-Recall曲线：更适合高度不平衡数据
 - F1-score：精确率和召回率的调和平均
3. 模型选择：
 - XGBoost：内置处理不平衡数据的能力（scale_pos_weight参数）
 - 随机森林：通过class_weight参数处理不平衡

模型训练最佳实践
 - 交叉验证：避免过拟合，提供更可靠的性能估计
 - 特征重要性：理解模型决策依据，指导业务决策
 - Pipeline集成：将预处理和模型训练整合为完整流程