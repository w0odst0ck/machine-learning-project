import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# 读取数据集
data = pd.read_csv('star_classification_add_normalized_dataset.csv')

# 提取特征和目标变量
X = data.iloc[:, :-1]  # 特征向量
y = data.iloc[:, -1]  # 目标变量

# 将目标变量转换为离散型
y = y.astype('category')

# 创建随机森林分类器
rf = RandomForestClassifier()

# 拟合模型并计算特征重要性
rf.fit(X, y)
feature_importances = rf.feature_importances_

# 创建特征重要性DataFrame
importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})

# 按特征重要性降序排序
importance_df = importance_df.sort_values('Importance', ascending=False)

# 打印特征重要性排序结果
print(importance_df)
