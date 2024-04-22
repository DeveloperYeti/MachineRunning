from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

X = np.array([[50,74] , [65,75] , [75,80], [80,82] , [95,85]])


plt.scatter(X[:,0], X[:,1])
plt.xlabel('X1')
plt.ylabel('X2')
plt.xlim(0,100)
plt.ylim(0,100)
# plt.show()

# print(np.mean(X,axis = 0 ))
# print(np.var(X,axis =0))

scalerX= StandardScaler()
scalerX.fit(X)
x_std = scalerX.transform(X)
# print(x_std)

# plt.scatter(x_std[:,0], x_std[:,1])
# plt.xlabel('X1')
# plt.ylabel('X2')
# plt.xlim(-3,3)
# plt.ylim((-3,3))
# plt.show()

print(np.mean(x_std , axis = 0))
print(np.cov(x_std[:,0], x_std[:,1] , ddof=0))

pca = PCA(n_components=2)
pca.fit(x_std)

print(pca.explained_variance_)
print(pca.explained_variance_ratio_)

Z = pca.transform(x_std)
plt.scatter(Z[:,0] , Z[:,1])
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.xlim(-2.5,-2.5)
plt.ylim(-2.5,-2.5)