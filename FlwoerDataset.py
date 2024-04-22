from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

iris = load_iris()
IrisData = iris.data[:5]

iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['targe'] = pd.Series(iris.target)
print(iris_df.head())
pd.options.display.max_columns = 100
print(iris_df.describe())
print(iris_df.info())

X=iris_df.iloc[:,:4]
y=iris_df.iloc[:,-1]

def iris_knn(X,y,k):
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)
    y_pred = knn.predict(X_test)
    return metrics.accuracy_score(y_test, y_pred)
train_test_split(X,y,random_state=42)

k=3
scores = iris_knn(X,y,k)
print('n_neighbors가 {0:d}일때 정확도: {1:3f}'.format(k,scores))



k=1
scores1 = iris_knn(X,y,k)
print(scores1)
k=5
scores2 = iris_knn(X,y,k)
print(scores2)
k=10
scores3 = iris_knn(X,y,k)
print(scores3)
k=20
scores4 = iris_knn(X,y,k)
print(scores4)
k=30
scores5 = iris_knn(X,y,k)
print(scores5)

iris =load_iris()
k=3
knn= KNeighborsClassifier(n_neighbors=k)
knn.fit(iris.data, iris.target)

classes ={0:'setosa',1:'versicolor',2:'virginica'}

X= [[4,2,1.3,0,4],
    [4,3,3.2,2.2]]
print('{}특성을 가지는 품종: {}'.format(X[0], classes[y[0]]))
print('{}특성을 가지는 품종: {}'.format(X[1], classes[y[1]]))

y_pred_all = knn.predict(iris.data)
scores = metrics.accuracy_score(iris.target, y_pred_all)
print('n_neighbors가 {0:d}일때 정확도: {1:.3f}'.format(k,scores))

plt.hist2d(iris.target, y_pred_all, bins=(3,3), cmap=plt.cm.jet)
plt.show()
plt.hist2d(iris.target, y_pred_all, bins=(3,3), cmap=plt.cm.gray)
plt.show()
#confusion_metrix -> 혼동 행렬 이라는 이름.

conf_mat = confusion_matrix(iris.target, y_pred_all)
conf_mat


