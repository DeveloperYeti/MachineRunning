import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import cluster

# 닥스훈트의 길이와 높이 데이터
dach_length =[77,78,85,83,73,77,73,80]
dach_height = [25,28,29,30,21,22,17,35]

# 사모예드의 길이와 높이 데이터
samo_length = [75,77,86,86,79,83,83,88]
samo_height = [56,57,50,53,60,53,49,61]

newdata_length = [79]
newdata_height = [35]

d_data = np.column_stack((dach_length,dach_height))
d_label =np.zeros(len(d_data))

s_data = np.column_stack((samo_length,samo_height))
s_label =np.ones(len(d_data))


#사모예드와 닥스 훈투의 길이. 높이 데이터는 중복되어 생략한다.
#개의 길이와 높이를 각각 ndarray형태로 만든다.
dogs = np.concatenate((d_data, s_data))
labels = np.concatenate((d_label, s_label))

dog_length = np.array(dach_length + samo_length)
dog_height = np.array(dach_height + samo_height)

dog_data = np.column_stack((dog_length,dog_height))

plt.title("Dog Data without label")
plt.scatter(dog_length,dog_height)
plt.show()

def kemeans_predcit_plot(x,k) :
    model = cluster.KMeans(n_clusters=k)
    model.fit(x)
    labels = model.predict(x)
    colors = np.array(['red', 'green', 'blue', 'magenta'])
    plt.suptitle('k-means clustering , k = {}'.format(k))
    plt.scatter(x[:,0], x[:, 1], color=colors[labels])
    plt.show()
kemeans_predcit_plot(dog_data , k =2)
kemeans_predcit_plot(dog_data , k =3)
kemeans_predcit_plot(dog_data , k =4)


newdata = [[82,40]]

dog_classes = {0:'Dachshound', 1:'Saymoyed'}

k = [1, 5, 9]
for i in range(len(k)):
    knn = KNeighborsClassifier(n_neighbors=k[i])
    knn.fit(dogs, labels)
    y_pred = knn.predict(newdata)
    print('k=', k[i], '일 때', newdata , ', 판정 결과:', dog_classes[y_pred[0]])






plt.scatter(dach_length, dach_height, c='red', label='dachshund')
plt.scatter(samo_length, samo_height, c='blue', marker='^', label='samoshund')
plt.scatter(newdata_length, newdata_height, s=100, marker='p', label='newdata', c='green')

plt.xlabel('Length')
plt.ylabel('Height')
plt.title('DogSize')
plt.legend(loc='upper left')

plt.show()

# 지도 학습.