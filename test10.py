from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터셋 불러오기
data = datasets.load_iris()
df = pd.DataFrame(data['data'], columns=data['feature_names'])
df['target'] = data['target']

# 데이터 확인
print("Dataset Summary:")
print(df.describe())

# 피처 스케일링
scalerX = StandardScaler()
X_scaled = scalerX.fit_transform(df.drop('target', axis=1))

# 공분산 행렬 계산
cov_matrix = np.cov(X_scaled.T)

# 고윳값과 고유 벡터 계산
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# 주성분 추출
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

# 주성분으로 데이터 변환
X_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2', 'PC3'])
X_df['target'] = df['target']

# KNN을 사용한 분류
knn = KNeighborsClassifier(n_neighbors=3)

# 학습용 데이터와 테스트용 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X_pca, df['target'], test_size=0.3, random_state=42)

# KNN 모델 학습
knn.fit(X_train, y_train)

# 테스트 데이터 예측
y_pred = knn.predict(X_test)

# 정확도 계산
accuracy = accuracy_score(y_test, y_pred)
print("KNN 분류 정확도:", accuracy)

# 주성분의 설명된 분산 비율
explained_variance_ratio = pca.explained_variance_ratio_
print("주성분의 설명된 분산 비율:", explained_variance_ratio)

# Scree Plot 그리기
plt.plot(np.arange(1, len(explained_variance_ratio) + 1), explained_variance_ratio, 'o-')
plt.title('Scree Plot')
plt.xlabel('주성분')
plt.ylabel('설명된 분산 비율')
plt.show()

# 주성분 로딩 플롯 그리기
loadings = pca.components_

plt.scatter(loadings[0,:], loadings[1,:], color='white')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.xlim(-0.5, 0.5)
plt.ylim(-0.5, 0.5)

for i in range(len(data.feature_names)):
    plt.arrow(0, 0, loadings[0, i], loadings[1, i], color='r', alpha=0.5)
    plt.text(loadings[0, i]*1.1, loadings[1, i]*1.1, data.feature_names[i], color='g', ha='center', va='center')

plt.title('주성분 로딩 플롯')
plt.show()