##
class Perceptron:

    def __init__(self,learning_rate = 0.01,n_iters=100):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.activation = self.step_func
        self.weights = None
        self.bias = None

    def step_func(self,x):
        return np.where(x>=0,1,0)


    def fit(self,X,y):
        n_samples,n_features = X.shape

        #initial weights
        self.weights = np.zeros(n_features)
        self.bias = 0

        y_ = [1 if i>0 else 0 for i in y]
        for _ in range(self.n_iters):
            for idx,x_i in enumerate(X):
                y_pred = self.predict(x_i)
                update = self.learning_rate * (y_[idx] -y_pred)
                self.weights += update * x_i
                self.bias += update

    def predict(self,X):
        line_op = np.dot(X,self.weights) + self.bias
        y_preds = self.activation(line_op)
        return y_preds
##

from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# from naive_bayes import NaiveBayes

def accuracy(y_true, y_pred):
    acc = np.sum(y_true == y_pred) / len(y_pred)
    return acc


X, y = datasets.make_blobs(n_samples=1000, n_features=2,centers=2,cluster_std=1.05, random_state=123)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

p  = Perceptron()
p.fit(X_train, y_train)
preds = p.predict(X_test)
# print(preds)
acc = accuracy(y_test, preds)
print(f"Accuracy : {acc}")
##

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
plt.scatter(X_train[:,0],X_train[:,1],marker='+',c=y_train)

x0_1 = np.amin(X_train[:,0])

x0_2 = np.amax(X_train[:,0])

##
x1_1 = (-p.weights[1]*x0_1 - p.bias)/p.weights[0]
x1_2 =(-p.weights[1]*x0_2 - p.bias)/p.weights[0]
##
ax.plot([x0_1,x0_2],[x1_1,x1_2],'k')

ymin = np.amin(X_train[:,1])
ymax = np.amax(X_train[:,1])

ax.set_ylim([ymin-1,ymax+3])
plt.show()

