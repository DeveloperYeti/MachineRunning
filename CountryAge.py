from statistics import LinearRegression

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
data_loc = 'https://github.com/dknife/ML/raw/main/data/'
life = pd.read_csv(data_loc + 'life_expectancy.csv')
life.head()
pd.set_option('display.max_seq_items', None)
#print(life)

life = life[['Life expectancy','Year', 'Alcohol',
            'Percentage expenditure' ,'Total expenditure',
            'Hepatitis B', 'Measles', 'Polio', 'BMI','GDP',
            'Thinness 1-19 years', 'Thinness 5-9 years']]
#print(life)
#print(life.shape)
#print(life.isnull().sum())
# life.dropna -> 실행이 될 경우 원본 데이터가 수정이 된다.
life.dropna(inplace = True)
print(life.shape)

sns.set(rc={'figure.figsize':(12,10)})# 상관 행렬 가시
correlation_matrix = life.corr().round(2) # 상관 행렬 생성
#sns.heatmap(data = correlation_matrix, annot=True)
# plt.show()# colab 등 노트북 환겨엥서는 필요없지만, 콘솔 환경 등에서는 필요.
X = life[['Alcohol', 'Percentage expenditure', 'Polio',
          'BMI','GDP','Thinness 1-19 years']]
y = life['Life expectancy']
# print(X)
# print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

lin_model = LinearRegression()
lin_model.fit(X_train, y_train)

y_hat_train = lin_model.predict(X_train)
plt.scatter(y_train, y_hat_train)
xy_range = [40,100]
plt.plot(xy_range, xy_range)

y_hat_test = lin_model.predict(X_test)
plt.scatter(y_test, y_hat_test)
plt.plot(xy_range, xy_range)

print('Mean squared error', mean_squared_error(y_test , y_hat_test))

n_X = normalize(X,axis=0)

nXtrain,nXtest, y_train, y_test = train_test_split(n_X,y,test_size=0.2)
lin_model.fit(nXtrain,y_train)

y_hat_train = lin_model.predict(nXtrain)
y_hat_test = lin_model.predict(nXtest)
plt.scatter(y_train, y_hat_train , color = 'r')
plt.scatter(y_test, y_hat_test , color = 'b')
plt.plot(xy_range, xy_range)


print('Mean squared error', mean_squared_error(y_test , y_hat_test))

# 일반화 시킨 것 원복시키는 방법

scaler = StandardScaler()
s_X = scaler.fit_transform(X)

plt.hist(s_X, bins=5)
plt.show()
