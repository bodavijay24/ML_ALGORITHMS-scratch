##
import numpy as np
class LinearRegression:
    def __init__(self,learning_rate=0.01,n_iters=100):
        self.lr = learning_rate
        self.epochs = n_iters
        self.w=None
        self.b =None

    def fit(self,X,y):
        n_samples,n_features  =X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        #gradient descent
        for i in range(self.epochs):
            print(f"Epochs {i+1}/{self.epochs}")
            y_hat = np.dot(X,self.w) + self.b
            dw = (1/n_samples) * np.dot(X.T,(y_hat-y))
            db = (1 / n_samples) * np.sum(y_hat - y)

            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict(self,X):
        return np.dot(X,self.w)+self.b



##
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import datasets

X,y = datasets.make_regression(n_samples=1000,n_features=1,noise=20,random_state=4)

X_train,X_test, y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=123)
##
def mse(y,y_hat):
    return np.mean((y-y_hat)**2)

##
linear = LinearRegression(learning_rate=0.01)
linear.fit(X_train,y_train)

preds = linear.predict(X_test)

print(f" MSE {mse(y_test,preds)}")
