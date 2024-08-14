# Experience Years: 근속연수, Salary:연봉($)
# (1) 주어진 데이터 셋에 대한 정보 파악(데이터 요약 및 시각화)
# (2) 근속연수에 따른 연봉에 관한 선형회귀 모형을 개발
# (3) 개발한 모델의 성능 평가(MSE, RMSEM, MAE)
# (4)모델이 예측한 값과 실제 값을 시각화 자료를 통해 비교(실제 값은 산점도로 표현, 예측 값은 선그래프로 표현)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error


data = pd.read_csv('./data/1.salary.csv',)

# 데이터 준비
X = data[['Experience Years']]
y = data['Salary']

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 모델 학습
model = LinearRegression()
model.fit(X_train, y_train)


# 예측
y_pred = model.predict(X_test)

# 성능 평가
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)

print(f'MSE: {mse}')
print(f'RMSE: {rmse}')
print(f'MAE: {mae}')

plt.figure(figsize=(10, 6))

# 실제 값 산점도
plt.scatter(X_test, y_test, color='blue', label='Actual Values')

# 예측 값 선 그래프
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted Values')

plt.xlabel('Experience Years')
plt.ylabel('Salary')
plt.title('Actual vs Predicted Salary')
plt.legend()
plt.show()