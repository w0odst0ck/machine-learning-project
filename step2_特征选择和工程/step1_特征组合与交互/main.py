import pandas as pd

# 读取数据集
data = pd.read_csv('star_classification_noclass异常值处理.csv')

# 特征差异
data['r_minus_g'] = data['r'] - data['g']

# 特征比率
data['r_divided_by_g'] = data['r'] / data['g']

# 特征乘积
data['r_times_g'] = data['r'] * data['g']

# 统计特征
data['mean'] = data.mean(axis=1)
data['std'] = data.std(axis=1)
data['max'] = data.max(axis=1)
data['min'] = data.min(axis=1)

# 交互特征
data['interaction_feature'] = (data['r'] - data['g']) * (data['u'] - data['z'])

# 保存结果到新的.csv文件
data.to_csv('star_classification_add.csv', index=False)
