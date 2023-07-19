import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# 读取数据集
data = pd.read_csv('your_data.csv')

# 提取特征向量
X = data.iloc[:, :-1]  # 特征向量

# 特征缩放
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 进行PCA降维
pca = PCA(n_components=9)
X_pca = pca.fit_transform(X_scaled)

# 创建随机森林分类器
classifier = RandomForestClassifier(max_depth=None, n_estimators=250)

# 拟合模型
classifier.fit(X_pca)

# 进行预测并打印分类结果
y_pred = classifier.predict(X_pca)
print("Class:", y_pred)
