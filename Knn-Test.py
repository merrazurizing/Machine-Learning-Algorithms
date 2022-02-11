import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

iris = datasets.load_iris()
X, y = iris.data,iris.target

X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=1234)

# print(X_train.shape)
# print(X_train[0])
# print(y_train.shape)
# print(y_train)

# plt.figure()
# plt.scatter(X[:,0],X[:,1],c=y,edgecolors='k',s=20)
# plt.show()

from Knn import KNN
#Clasificador
clf= KNN(k=5)
clf.fit(X_train,y_train)
predictions = clf.predict(X_test)

#Calcular precision , por cada prediccion igual a la categoria correcta añade 1 y luego divido por num de test
acc = np.sum(predictions == y_test) / len(y_test) *100
print(acc)