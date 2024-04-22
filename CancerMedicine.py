from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = load_breast_cancer(as_frame = True)
print(data.frame)

print(data.data)
print(data.target)
print(data.DESCR)
print(data.feature_names)
print(data.target_names)

data_mean = data.frame[['mean radius','mean texture', 'mean perimeter','mean area','target']]
sns.pairplot(data_mean , hue = 'target')
plt.show()

X_train,X_test,Y_train,Y_test = train_test_split(data.data,data.target,
                                                 test_size=0.3 ,random_state = 1234)

print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

scalerX =StandardScaler()
scalerX.fit(X_train)
X_train_std = scalerX.transform(X_train)
print(X_train_std)

X_test_std = scalerX.transform(X_test)
print(X_test_std)

## 최근접 이웃 수 결정
# 학습용 데이터의 분류 정확도
train_accuracy = []
#테스트 데이터의 분류 정확도
test_accuracy = []

# 최근접 이웃의 수 : 1~15
neighbors = range(1,16)
for k in neighbors:
    #모형화
    knn = KNeighborsClassifier(n_neighbors=k)
    #학습
    knn.fit(X_train_std,Y_train)
    #학습 데이터의 분류 정확도
    score = knn.score(X_train_std,Y_train)
    train_accuracy.append(score)
    #테스트 데이터의 분류 정확도
    score = knn.score(X_test_std,Y_test)
    test_accuracy.append(score)

#K의 크기에 따른 분류 정확도 변화
plt.plot(neighbors,train_accuracy,label='train')
plt.plot(neighbors,test_accuracy, label='test')
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.legend()
plt.show()



