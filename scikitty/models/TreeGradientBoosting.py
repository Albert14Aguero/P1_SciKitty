import numpy as np
import pandas as pd
from models.DecisionTree import DecisionTree
from sklearn.preprocessing import OneHotEncoder

class Stump(DecisionTree):
    def __init__(self, min_samples=1, max_depth=1, gini=False):
        super().__init__(min_samples=min_samples, max_depth=max_depth, gini=gini)

def residual(y, y_pred):
    return y - y_pred

def decay(alpha, alpha_decay=0.9):
    return alpha * alpha_decay

class tree_gradient_boosting:
    def __init__(self, alpha=0.2, T=100, criterion='gini', alpha_min=0.01, alpha_decay=0.99):
        self.alpha = alpha
        self.T = T
        self.criterion = criterion
        self.alpha_min = alpha_min
        self.alpha_decay = alpha_decay
        self.models = []
        self.alphas = []

    def fit(self, X_train, y_train):

        r = y_train.copy()
        h = np.zeros_like(y_train, dtype=float)

        for i in range(self.T):
            if self.criterion == 'gini':
                final_crit = True
            else:
                final_crit = False

            tree = Stump(min_samples=1, max_depth=1, gini=final_crit)
            tree.fit(X_train, r)

            self.models.append(tree)
            self.alphas.append(self.alpha)

            h += self.alpha * tree.predict(X_train)
            r = residual(y_train, h)

            self.alpha = decay(self.alpha, self.alpha_decay)

            if self.alpha < self.alpha_min:
                break

    def predict(self, X_test):

        h = np.zeros((X_test.shape[0],), dtype=float)
        print(len(self.models))
        for model, alpha in zip(self.models, self.alphas):
            
            h += alpha * model.predict(X_test)
            
        return (h > 0.5).astype(int)
