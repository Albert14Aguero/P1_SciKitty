import numpy as np


class LinearRegressionP:
    def __init__(self, learning_rate = 0.005, num_iters = 500):
        self.weights = None
        self.bias = None
        self.learning_rate = learning_rate
        self.num_iters = num_iters

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.bias = 0
        self.weights = np.zeros(num_features)

        for _ in range(self.num_iters):
            y_predict = np.dot(X, self.weights) + self.bias

            dw = (2/num_samples) * np.dot(X.T, (y_predict - y))
            db = (2/num_samples) * np.sum(y_predict - y)

            self.weights = self.weights - self.learning_rate  * dw
            self.bias = self.bias - self.learning_rate  * db
        
    def predict(self, X):
        y_predict = np.dot(X, self.weights) + self.bias
        return y_predict
        