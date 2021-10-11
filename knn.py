##
from collections import Counter
import numpy as np
def eucledian(x1,x2):
    return np.sqrt(np.sum((x1-x2)**2))

class KNN:
    def __init__(self,k=3):
        self.k = k

    def fit(self,X,y):
        self.X = X
        self.y = y

    def predict(self,X_test):
        preds = [self._predict(x) for x in X_test]
        return np.array(preds)

    def _predict(self,x):
        distances = [eucledian(x,x_tr) for x_tr in self.X]
        print(distances)
        k_idx = np.argsort(distances)[:self.k]
        k_nearest = [self.y[i] for i in k_idx]

        most_common = Counter(k_nearest).most_common(1)
        return most_common[0][0]

##

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import datasets

iris = datasets.load_iris()
X,y = iris.data,iris.target

X_train,X_test, y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=123)

knn = KNN(4)
knn.fit(X_train,y_train)
preds = knn.predict(X_test)
acc  = np.sum(preds==y_test)/len(y_test)
print(f"Accuracy {acc}")


