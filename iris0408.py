import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# Iris 데이터 로드
data = load_iris(as_frame=True)
iris = pd.DataFrame(data.data)
iris['target'] = data.target

# 결측치 제거
iris.dropna(inplace=True)


X = iris[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']]
y = iris['target']

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# 특성 데이터 표준화
scalerX = StandardScaler()
X_train_std = scalerX.fit_transform(X_train)
X_test_std = scalerX.transform(X_test)


# KNN 분류 모델 학습
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_std, y_train)

# 테스트 데이터에 대한 예측
y_pred = knn.predict(X_test_std)

# 정확도 계산 및 출력
accuracy = accuracy_score(y_test, y_pred)
print("KNN 분류 정확도:", accuracy)

best_model = {"k":0,"score":0.0}
## 최근접 이웃 수 결정
# 학습용 데이터의 분류 정확도
train_accuracy = []
# 테스트 데이터의 분류 정확도
test_accuracy = []

# 최근접 이웃의 수 : 1~15
neighbors = range(1, 16)
for k in neighbors:
    # 모형화
    knn = KNeighborsClassifier(n_neighbors=k)
    # 학습
    knn.fit(X_train_std, y_train)
    # 학습 데이터의 분류 정확도
    score = knn.score(X_train_std, y_train)
    train_accuracy.append(score)
    # 테스트 데이터의 분류 정확도
    score = knn.score(X_test_std, y_test)
    test_accuracy.append(score)
    if best_model["score"] < score:
        best_model["k"] = k
        best_model["score"] = score
print(best_model)


# K의 크기에 따른 분류 정확도 변화
plt.plot(neighbors, train_accuracy, label='train')
plt.plot(neighbors, test_accuracy, label='test')
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('plot.png')
plt.show()


cm = confusion_matrix(y_test, y_pred)


plt.figure(figsize=(8, 6))
sns.heatmap(data=cm, annot=True, cmap='Blues', fmt='g')
plt.title('Confusion Matrix')
plt.xlabel('Predict')
plt.ylabel('True Label')
plt.savefig('Confusion Matrix.png')
plt.show()
print(cm)