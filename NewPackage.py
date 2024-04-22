import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

#학습용 데이터
# 입력
X_train = np.array([[25,25],
                    [33,30],
                    [38,30],
                    [45,35],
                    [28,40]])


y_train = np.array([0,0,1,1,0])

X_test = np.array([[30,35]])

plt.scatter(X_train[:,0],X_train[:,1] ,c = y_train)
plt.scatter(X_test[:,0] , X_test[:,1],c = 'red', marker = 'D' , s=100 )
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()
# test 데이터는 피팅하면 안됨 테스트를 위한 무언가를 해서는 안됨.
scalerX = StandardScaler()
scalerX.fit(X_train)
X_train = scalerX.transform(X_train)
print(X_train)

X_test_std = scalerX.transform(X_test)
print(X_test_std)



#모형화
knn = KNeighborsClassifier(n_neighbors= 3 , metric='euclidean')
#학습
knn.fit(X_train,y_train)

#예측
pred = knn.predict(X_test_std)
print(pred)

#클래스별 확률 값을 반환
knn.predict(X_test_std)
