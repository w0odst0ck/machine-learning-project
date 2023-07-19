import pandas as pd
from sklearn.model_selection import train_test_split

# 读取原始数据集的CSV文件
data = pd.read_csv('star_classification_add_3sigma_异常值替换为零_归一化_class.csv')

# 划分数据集为训练集、验证集和测试集
train_data, temp_data = train_test_split(data, test_size=0.3, random_state=42)  # 将数据划分为训练集和临时数据集
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)  # 将临时数据集划分为验证集和测试集

# 将划分后的数据保存为CSV文件
train_data.to_csv('train_dataset.csv', index=False)
val_data.to_csv('val_dataset.csv', index=False)
test_data.to_csv('test_dataset.csv', index=False)
