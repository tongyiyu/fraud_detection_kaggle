# validate_submission.py
import pandas as pd

def validate_submission():
    """验证提交文件格式"""
    submission = pd.read_csv('submissions/submission_v1.csv')
    
    # 检查基本要求
    required_columns = ['id', 'fraud_probability']
    assert all(col in submission.columns for col in required_columns), "缺少必需的列"
    
    # 检查数据类型
    assert pd.api.types.is_numeric_dtype(submission['fraud_probability']), "fraud_probability必须是数值类型"
    
    # 检查概率范围
    assert submission['fraud_probability'].min() >= 0, "概率不能小于0"
    assert submission['fraud_probability'].max() <= 1, "概率不能大于1"
    
    # 检查行数
    test_df = pd.read_csv('data/test.csv')
    assert len(submission) == len(test_df), "提交文件行数与测试集不匹配"
    
    print("✅ 提交文件格式验证通过！")
    print(f"提交文件形状: {submission.shape}")
    print(f"欺诈概率统计:\n{submission['fraud_probability'].describe()}")

if __name__ == "__main__":
    validate_submission()