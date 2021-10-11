import numpy as np

class SVM:

    def __init__(self,learning_rate = 0.01,lambda_p = 0.01,n_iters = 100):

        self.lr = learning_rate
        self.lamda =lambda_p
        self.n_iters = n_iters
        self.w = None
        self.b  =None

    def fit(self,X,y):
        y_ = np.where(y<=0,-1,1)
        n_samples,n_features = X.shape

        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters)

            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i,self.w)-self.b)>=1
                if condition:
                    self.w -= self.lr * (2*self.lamda*self.w)
                else:
                    self.w -= self.lr * (2 * self.lamda * self.w)-np.dot(x_i,y_[idx])
        pass

    def predict(self,X):
        lop = np.dot(X,self.w)-b
        return np.sign(lop)
