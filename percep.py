#!/usr/bin/env python
# coding: utf-8

# In[1]:

import pickle
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


iris = load_iris()
df = pd.DataFrame(data = np.c_[iris['data'], iris['target']], 
                  columns=iris['feature_names'] + ['target'])
X = df.iloc[:, [0, 2]]
y = df.iloc[:, [4]]

class Perceptron:
    
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter
        self.w_ = None
        self.errors_ = None
    
    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []
        
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
    
    def predict(self, X):
        return np.where(self._net_input(X) >= 0, 1, -1)
    
    def _net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

finalp = Perceptron(0.01, 1000)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train = X_train.to_numpy()
y_train = y_train.to_numpy()
finalp.fit(X_train, y_train)

with open('p.pkl', 'wb') as m:
    pickle.dump(finalp, m)


# In[ ]:




