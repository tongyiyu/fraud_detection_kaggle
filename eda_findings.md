# 保险反欺诈数据集 EDA发现

## 📊 数据概览
- **样本数量**: 30,000+
- **特征数量**: 20+ 个
- **目标变量**: `fraud` (0=正常, 1=欺诈)
- **欺诈比例**: ~5% (典型的不平衡数据集)

## 🔍 关键发现
1. **缺失值情况**:
   - `claim_amount`: 无缺失
   - `policy_tenure`: 少量缺失 (~2%)
   - `witness_present_ind`: 较多缺失 (~15%)

2. **重要特征**:
   - `claim_amount`: 欺诈案例的索赔金额明显更高
   - `age_of_driver`: 年轻司机欺诈率较高
   - `past_num_of_claims`: 历史索赔次数与欺诈正相关

3. **数据质量问题**:
   - 部分分类变量存在拼写不一致
   - 日期格式需要标准化
   - 极端值（异常高的索赔金额）

## 📈 可视化洞察
![目标分布](visualizations/target_distribution.png)
- 严重的类别不平衡，需要特殊处理

![数值特征](visualizations/numeric_features.png)
- `claim_amount` 在欺诈案例中右偏更严重

![相关性](visualizations/correlation_heatmap.png)
- `claim_amount` 与 `fraud` 相关性最高 (0.35)