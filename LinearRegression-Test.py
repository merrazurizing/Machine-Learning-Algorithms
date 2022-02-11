import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

X, y = datasets.make_regression(n_samples=100 , n_features=1 , noise=20 , random_state=4)

X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=1234)

from LinearRegression import LinearRegression
regressor = LinearRegression(lr=0.01)
regressor.fit(X_train , y_train)
predicted = regressor.predict(X_test)

#Mean square error , muestra que grande es la diferencia del valor real y el predecido
def mse(y_true , y_predicted):
    return np.mean((y_true - y_predicted)**2)

mse_value=mse(y_test,predicted)
print(mse_value)

y_pred_line = regressor.predict(X)
fig = plt.figure(figsize=(8, 6))
m1 = plt.scatter(X_train, y_train, s=10)
m2 = plt.scatter(X_test, y_test, s=10)
plt.plot(X, y_pred_line, color="black", linewidth=2, label="Prediction")
plt.show()

