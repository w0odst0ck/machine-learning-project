import pandas as pd
import numpy as np

# 读取数据集
data = pd.read_csv('star_classification_no_class.csv')

# 处理缺失值
data.fillna(0, inplace=True)

# 识别并处理异常值
numeric_columns = data.select_dtypes(include=np.number).columns
for column in numeric_columns:
    mean = data[column].mean()
    std = data[column].std()
    threshold = 3 * std  # 设置异常值的阈值
    lower_bound = mean - threshold
    upper_bound = mean + threshold
    data[column] = np.where((data[column] < lower_bound) | (data[column] > upper_bound), 0, data[column])

# 将处理后的数据保存为.csv文件
data.to_csv('star_classification_noclass异常值处理.csv', index=False)
