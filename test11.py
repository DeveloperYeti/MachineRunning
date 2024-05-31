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



data = datasets.load_breast_cancer()

df = pd.DataFrame(data['data'], columns=data['feature_names'])
df['target'] = data['target']

# plt.figure(figsize=(8, 8))
# ax = sns.pairplot(df, hue='target')
# plt.show()


scalerX = StandardScaler()
scalerX.fit(data.data)
X_std = scalerX.transform(data.data)
print(X_std)

pca = PCA()
pca.fit(X_std)

print(pca.explained_variance_)
print(pca.explained_variance_ratio_)

plt.plot(pca.explained_variance_ratio_ , 'o-')
plt.title('ScreePlot')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance')
plt.show()

component = 30

pca = PCA(n_components= component)
Z = pca.fit_transform(X_std)

Z_df = pd.DataFrame(data=Z[:, :component], columns=[f'PC{i}' for i in range(1, component + 1)])
print(Z_df.head())

# loadings = pca.components_
#
# rows, columns = loadings.shape
#
# rows_names = ['PC1', 'PC2', 'PC3']  # loadings 배열의 행 수에 맞게 수정
# for i in range(rows):
#     plt.arrow(0, 0, loadings[i, 0], loadings[i, 1], color='r', alpha=0.5)
#     plt.text(loadings[i, 0] * 1.2, loadings[i, 1] * 1.2, rows_names[i], color="g",
#              ha='center', va='center')
# plt.savefig('diemensionality reduction.png')
# plt.show()

cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)



# Elbow point 찾기
elbow_point = np.where(cumulative_variance_ratio < 0.5)[0]

if len(elbow_point) > 0:
    elbow_point = elbow_point[-1] + 1
    print("Elbow point (0.5 미만인 주성분 개수):", elbow_point)
    print("Elbow point 값 : ", cumulative_variance_ratio[elbow_point - 1])
else:
    print("주의: 설명 분산이 0.5 미만인 주성분을 찾지 못했습니다.")
print()



X_pca = Z
    # 데이터 분할

scalerX.fit(X_pca)
X_train_std = scalerX.transform(X_pca)
X_test_std = scalerX.transform(X_pca)

for i in reversed(range(0, 4)):
    X_train, X_test, y_train, y_test = train_test_split(X_pca, df['target'], test_size=0.3, random_state=i)

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)

    best_model = {"k": 0, "score": 0.0}
    ## 최근접 이웃 수 결정
    # 학습용 데이터의 분류 정확도
    train_accuracy = []
    test_accuracy = []

    # 최근접 이웃의 수 : 1~15
    neighbors = range(2, 16)
    for k in neighbors:
        # 모형화
        knn = KNeighborsClassifier(n_neighbors=k)
        # 학습
        knn.fit(X_train, y_train)
        # 학습 데이터의 분류 정확도
        score = knn.score(X_train, y_train)
        train_accuracy.append(score)
        # 테스트 데이터의 분류 정확도
        score = knn.score(X_test, y_test)  # 여기서 수정
        test_accuracy.append(score)
        if best_model["score"] < score:
            best_model["k"] = k
            best_model["score"] = score

    print(best_model)

    plt.figure(figsize=(16, 8))
    plt.plot(neighbors, train_accuracy, label='train')
    plt.plot(neighbors, test_accuracy, label='test')
    plt.title(f"random state{i}")
    plt.xlabel('K')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f'knn_components_Original_random_state_{i}.png')  # 파일명에 랜덤 시드 추가
    plt.show()




#
#결과 변환된 z 데이터가 knn 에 넣엇을때 의사용
# PCA를 사용하여 데이터를 변환한 후에는 주어진 데이터를 새로운 축(주성분)으로 투영합니다. 이 때, 각 주성분은 원본 데이터의 특성들을 선형 결합하여 만들어진 것입니다.
#
# 예를 들어, 주성분(PC1)이 x와 y 특성의 선형 결합으로 정의된다고 가정해 보겠습니다. 그러면 PC1은 x와 y 특성의 가중치로 구성된 새로운 특성일 것입니다. 따라서 주어진 데이터를 PC1에 투영하면, 각 데이터 포인트는 x와 y 특성을 이용하여 PC1에 대응하는 값으로 변환됩니다.
#
# 이런 식으로, PCA를 통해 변환된 데이터인 Z는 주성분들의 조합으로 이루어진 새로운 특성 공간을 형성하게 됩니다. 따라서 KNN 분류기는 이 새로운 특성 공간에서 데이터를 분류하게 되며, 이 때 각 데이터 포인트의 이웃들의 클래스를 기준으로 분류를 수행합니다.
#
# 즉, PCA를 통해 변환된 Z 데이터를 KNN에 입력으로 제공하면, KNN은 Z 공간에서 각 데이터 포인트의 이웃들을 찾아서 다수결 또는 가중치 평균 등의 방법을 사용하여 해당 데이터 포인트의 클래스를 예측하게 됩니다.