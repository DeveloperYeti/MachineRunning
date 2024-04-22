from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns




data = load_iris(as_frame=True)
data_df = pd.DataFrame(data.data, columns=data.feature_names)
data_df['target'] = data.target

print(data_df['target'])
print()
sns.pairplot(data_df, vars = ['sepal width (cm)', 'petal length (cm)','petal width (cm)'],hue='target')
plt.show()

X_train,X_test,Y_train,Y_test = train_test_split(data.data,data.target,
                                                 test_size=0.3 ,random_state = 1234)

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
