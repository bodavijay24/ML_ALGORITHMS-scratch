import numpy as np
class LogisticRegression:

    def __init__(self,lr = 0.01,epochs = 100):
        self.lr = lr
        self.epochs = epochs
        self.w = None
        self.b = None

    def fit(self,X,y):
        n_samples,n_features = X.shape
        self.w = np.zeros(n_features)
        self.b  = 0

        for i in range(self.epochs):

            y_hat = self._sigmoid(np.dot(X,self.w) + self.b)
            dw = np.dot(X.T,y_hat-y)/n_samples
            db = np.sum(y_hat-y) / n_samples

            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict(self,X):
        lm = np.dot(X,self.w) + self.b
        y_pred = self._sigmoid(lm)
        y_hat = [1 if i>0.5 else 0 for i in y_pred]
        return  y_hat
    def _sigmoid(self,z):
        return 1/(1+np.exp(-z))


##
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import datasets

bc = datasets.load_breast_cancer()
X,y = bc.data,bc.target

X_train,X_test, y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=123)
##
linear = LogisticRegression(lr=0.0001,epochs=1000)
linear.fit(X_train,y_train)
preds = linear.predict(X_test)
acc = np.sum(preds==y_test)/len(y_test)

print(f" Accuracy {acc}")

