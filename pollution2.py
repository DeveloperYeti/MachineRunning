import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error as mse

data_home = 'https://github.com/dknife/ML/raw/main/data/'
lin_data = pd.read_csv(data_home + 'pollution.csv')

def h(x, w, b):
    return w * x + b

w, b = -3, 6
x0, x1 = 0.0, 0.5

lin_data.plot(kind = 'scatter', x = 'input', y = 'pollution')
plt.plot([x0, x1], [h(x0, w, b), h(x1, w, b)])
plt.show()
x = lin_data['input']
y = lin_data['pollution']

learning_rate = 0.0025

for i in range(10000):
    y_hat = h(x, w, b)
    error = y_hat -y # mse 를 안쓴이유는 이미 미분이 된 값이기 때문에.
    w = w - learning_rate * (error * x).sum()
    b = b - learning_rate * error.sum()

lin_data.plot(kind = 'scatter', x = 'input', y = 'pollution')
plt.plot([x0, x1], [h(x0, w, b), h(x1, w, b)])
plt.show()