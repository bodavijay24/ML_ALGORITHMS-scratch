#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 18:13:28 2021

@author: vijay
"""
import numpy as np


class NaiveBayes:
    def fit(self,X,y):
        
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        #Assume it follows gaussian distribution
        #initialize means, variance and prioirs
        print(n_classes,n_features)
        self._mean = np.zeros((n_classes,n_features),dtype = np.float64)
        self._var = np.zeros((n_classes,n_features),dtype = np.float64)
        self._prior = np.zeros(n_classes,dtype=np.float64)

        print("Training Started...\n")
        for c in self._classes:
            
            Xc = X[c==y]
            self._mean[c,:] = Xc.mean(axis=0)
            self._var[c,:] = Xc.var(axis=0)
            self._prior[c] = len(Xc)/float(n_samples)
        print("Awesome...! Training successfully finished...\n")
    
    def predict(self,X):
        y_preds = [self._predict(x) for x in X]
        return y_preds
    
    def _predict(self,x):
        posteriors = []
        
        for idx, c in enumerate(self._classes):
            
            prior = np.log(self._prior[idx])
            class_condi = np.sum(np.log(self._pdf(idx,x)))
            posterior = class_condi + prior
            posteriors.append(posterior)
        return self._classes[np.argmax(posteriors)]
            
            
    def _pdf(self,cidx,x):
        
        mean=self._mean[cidx]
        var = self._var[cidx]
        num = np.exp(-(x-mean)**2 / 2* var)
        den = np.sqrt(2 * np.pi * var)
        return num/den


##
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# from naive_bayes import NaiveBayes

def accuracy(y_true,y_pred):
    acc = np.sum(y_true==y_pred)/len(y_pred)
    return acc


X,y = datasets.make_classification(n_samples=1000, n_features=10,n_classes=2,random_state=123)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=123)

nb = NaiveBayes()
nb.fit(X_train,y_train)
preds = nb.predict(X_test)
# print(preds)
acc  = accuracy(y_test,preds)
print(f"Accuracy : {acc}")
##
        

        