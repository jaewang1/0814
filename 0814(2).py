import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import accuracy_score

data = pd.read_csv('./data/1.salary.csv')

array = data.values
array.shape
X = array[:, 0]
Y = array[:, 1]

XR = X.reshape(-1, 1)
# 데이터 분할
X_train, X_test, Y_train, Y_test = train_test_split(XR, Y, test_size=0.3, random_state=0)

# 모델 선택 및 학습
model = LinearRegression()
model.fit(X_train, Y_train)
model.coef_
model.intercept_
y_pred = model.predict(X_test)
error = mean_absolute_error(y_pred, Y_test)

plt.figure(figsize=(10, 6))
plt.scatter(range(len(X_test)), Y_test, color='blue', label='Actual Values', marker='o')
plt.plot(range(len(Y_test)), y_pred, color='red', label='Predicted Values', marker='x')

plt.title("Experience Years-Salary")
plt.xlabel("Experience Years")
plt.ylabel('Salary')
plt.legend()
plt.show()
plt.savefig('./results/scatter.png')
