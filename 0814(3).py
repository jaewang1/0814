import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import accuracy_score

header = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
data = pd.read_csv('./data/2.iris.csv', names=header)
array = data.values
array.shape
X = array[:, 0:4]
Y = array[:, 4]

# 데이터 분할
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)




# 모델 선택 및 학습
model = DecisionTreeClassifier(max_depth=1000, min_samples_split=50, min_samples_leaf=5)
fold = KFold(n_splits=10, shuffle=True)
acc = cross_val_score(model, X, Y, cv=fold, scoring='accuracy')
print(acc)
model.fit(X_train, Y_train)
y_pred = model.predict(X_test)

plt.figure(figsize=(10, 6))
plt.scatter(range(len(X_test)), Y_test, color='blue', label='Actual Values', marker='o')
plt.scatter(range(len(Y_test)), y_pred, color='red', label='Predicted Values', marker='x')

plt.title("iris")
plt.xlabel("Experience Years")
plt.ylabel('Salary')
plt.legend()
plt.show()
plt.savefig('./results/scatter.png')
