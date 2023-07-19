import pandas as pd

# 读取数据集
data = pd.read_csv('star_classification_add_normalized_dataset.csv')

# 计算特征向量之间的Pearson相关系数
correlation_matrix = data.corr(method='pearson')

# 打印相关系数矩阵
print(correlation_matrix)
