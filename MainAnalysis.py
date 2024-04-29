from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

x=np.array([[50,73],[65,75],[75,80],[80,82],[95,85]])

plt.scatter(x[:,0],x[:,1])
plt.xlabel('x1')
plt.ylabel('x2')
plt.xlim(0,100)
plt.ylim(0,100)
plt.show()

print(np.mean(x,axis=0))
print(np.var(x,axis=0))

scalerX = StandardScaler()
scalerX.fit(x)
x_std = scalerX.transform(x)
print(x_std)

plt.scatter(x_std[:,0],x_std[:,1])
plt.xlabel('x1')
plt.ylabel('x2')
plt.xlim(-3,3)
plt.ylim(-3,3)
plt.show()

print(np.mean(x_std,axis=0))
print(np.cov(x_std[:,0],x_std[:,1],ddof=0))

pca = PCA(n_components=2)
pca.fit(x_std)

print(pca.explained_variance_)
print(pca.explained_variance_ratio_) #pca1, pca2 를 통해 pca1이 엄청난 연관도가 있다.
z = pca.fit_transform(x_std)

plt.scatter(z[:,0],z[:,1])
plt.xlabel('pc1')
plt.ylabel('pc2')
plt.xlim(-2.5,2.5)
plt.ylim(-2.5,2.5)
plt.show()

loadings = pca.components_
print(loadings)

plt.scatter(loadings[:,0],loadings[:,1],color="w")
plt.xlabel('pc1')
plt.ylabel('pc2')
plt.xlim(-2,2)
plt.ylim(-2,2)

rows,columns = loadings.shape

rows_names = ['x1','x2']
for i in range(rows):
    plt.arrow(0,0,loadings[i,0],loadings[i,1],color='r',alpha=0.5)

    plt.text(loadings[i,0]*1.2,loadings[i,1]*1.2,rows_names[i],color="g",
             ha='center',va='center')
plt.show()