import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import numpy as np
iris = load_iris(as_frame=True)
# Iris 데이터셋을 pandas DataFrame으로 변환
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target
correlation_matrix = iris_df.corr().round(2)
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix of Iris Dataset')
plt.xlabel('Features')
plt.ylabel('Features')
plt.show()
X = iris_df[['sepal length (cm)', 'petal length (cm)', 'petal width (cm)', 'sepal width (cm)']]
y = iris.target
print(X)
print(y)
# 학습용과 테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)
# 피처 스케일링: 학습데이터
scalerX = StandardScaler()
scalerX.fit(X_train)
X_train_std = scalerX.transform(X_train)
print(X_train_std)
# 피처 스케일링: 테스트데이터
X_test_std = scalerX.transform(X_test)
print(X_test_std)
scalerX = StandardScaler()
scalerX.fit(X_train)
x_train_std = scalerX.transform(X_train)
x_test_std = scalerX.transform(X_test)
clf = svm.SVC(kernel='rbf',c=1 ,  gamma=0.02061731110582648 )
param_grid = {"gamma" : np.logspace(-10, 1 , 11 , base = 2) , "c": [0.5,1.0,2.0]}
grid_model = GridSearchCV(clf, param_grid = param_grid, scoring='param_grid', iid = False)
grid_model.fit(X_train_std, y_train)
print(grid_model.best_params_)
clf.fit(x_train_std, y_train)
y_pred = clf.predict(x_test_std, y_test)
clf.score(x_test_std, y_test)
cf = confusion_matrix(y_test, y_pred)
print(cf)
print(clf.score(x_test_std, y_test))
