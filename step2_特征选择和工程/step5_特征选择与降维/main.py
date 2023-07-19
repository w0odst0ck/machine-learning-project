import pandas as pd
import xgboost as xgb

# 读取数据集
data = pd.read_csv('star_classification_add_normalized_dataset_noclass.csv')

# 提取特征和目标变量
X = data.iloc[:, :-1]  # 特征向量
y = data.iloc[:, -1]  # 目标变量

# 创建XGBoost模型
model = xgb.XGBRegressor()

# 拟合模型
model.fit(X, y)

# 提取特征重要性
feature_importances = model.feature_importances_

# 创建特征重要性DataFrame
importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})

# 按特征重要性降序排序
importance_df = importance_df.sort_values('Importance', ascending=False)

# 选择前k个重要特征
k = 10  # 设置要选择的特征数量
selected_features = importance_df['Feature'][:k]

# 从原始数据集中提取选定的特征
selected_data = data[selected_features]

# 将选定的特征保存为CSV文件
selected_data.to_csv('selected_features.csv', index=False)
