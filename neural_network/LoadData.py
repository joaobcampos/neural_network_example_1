import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

class LoadData:
    
    def __init__(self):
        data = pd.read_csv('data/iris.csv')
        self.X = np.array(data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']])
        self.y = np.array(pd.get_dummies(data['species']))
    
    def normalize(self, X, flag):
        X_ = X.copy()
        X_ = np.array(X_)
        scaler = MinMaxScaler()
        return scaler.fit_transform(X_) if flag else X_
    
    def reshape_array(self, X):
        X_ = X.copy()
        X_ = X_.T
        X_ = np.reshape(X_, (X_.shape[0], 1, X_.shape[1]))
        return X_
 
    def partition_dataset(self, test_size = 0.33, normalize=True):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=42)
        if normalize == True:
            X_train = self.normalize(X_train, normalize)
            X_test  = self.normalize(X_test, normalize)
        X_train = self.reshape_array(X_train)
        X_test  = self.reshape_array(X_test)
        y_train = self.reshape_array(y_train)
        y_test  = self.reshape_array(y_test)
        return X_train, X_test, y_train, y_test