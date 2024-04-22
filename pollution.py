import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error

# 데이터 저장 위치

data_home = 'https://github.com/dknife/ML/raw/main/data/'
lin_data = pd.read_csv(data_home + 'pollution.csv')  # 데이터 파일 이름
print(lin_data)

# lin_data.plot(kind = 'scatter' , x = 'input', y= 'pollution')

# 두 변수들이 input을 독립변수 x로, pollution을 종속변수 y로 하는 y= ws +b 라는 직선으로 표현하면, 데이터가 이 함수를 따를 것이라느 가설 제시

w, b = 1, 1
x0, x1 = 0.0, 1.0


def h(x, w, b):
    return w * x + b


# 데이터 (산포도)와 가설(직선)을 비교

# lin_data.plot(kind = 'scatter' , x = 'input', y= 'pollution')
# plt.plot([x0,x1], [h(x0,w,b),h(x1,w,b)])

#y_hat = np.array([1.2, 2.1, 2.9, 4.1, 4.7, 6.3, 7.1, 7.7, 8.5, 10.1])
# y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# diff_square = (y_hat - y) ** 2
# e_mse = diff_square.sum() / len(y)
# print(e_mse)
#
# print('Mean squared error:', mse(y_hat, y))
#
# print('Mean absolute error:', mean_absolute_error(y_hat, y))

x = np.array([1, 4.5, 9, 10, 13])
y = np.array([0, 0.2, 2.5, 5.4, 7.3])

errors = []

w_list = np.arange(1.0, 0.2, -0.01)

for w in list(w_list):
    y_hat = w * x
    errors.append(mse(y_hat, y))
    print('w = {:.1f}, 평균 제곱 오차: {:.2f}'.format(w, mse(y_hat, y)))

plt.plot(w_list, errors)
plt.xlabel('w')
plt.ylabel('Mse')
plt.show()


