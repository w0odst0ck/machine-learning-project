import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 读取数据集
data = pd.read_csv('star_classification_add.csv')

# 提取特征列
features = data.iloc[:, :-1]

# 实例化归一化器
scaler = MinMaxScaler()

# 对特征进行归一化
normalized_features = scaler.fit_transform(features)

# 构建归一化后的数据集
normalized_data = pd.DataFrame(normalized_features, columns=features.columns)

# 将归一化后的数据集保存为.csv文件
normalized_data.to_csv('star_classification_add_normalized_dataset.csv', index=False)
