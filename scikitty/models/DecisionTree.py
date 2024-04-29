import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from graphviz import Digraph

class Node:
    def __init__(self, feature_index=None, impurity=None, impurity_type=None, samples = None, left=None, right=None, value=None, split_treshold = None):
        self.feature_index = feature_index
        self.impurity = impurity
        self.impurity_type = impurity_type
        self.samples = samples
        self.left = left
        self.right = right
        self.value = value
        self.split_treshold = split_treshold

class DecisionTree:
    def __init__(self, min_samples = 1, max_depth = 3, gini = False):
        self.root = None
        self.min_samples = min_samples
        self.max_depth = max_depth
        self.gini = gini
    
    def fit(self, X, Y):
        self.root = self.build_tree(X, Y)
    
    def build_tree(self, X, Y, depth=0):
        num_samples, num_features = np.shape(X)
        if num_samples >= self.min_samples and depth <= self.max_depth:
            
            best_split = self.best_split(X, Y)
            
            if best_split["impurity"] > 0:
                left_node = self.build_tree(best_split["x_left"], best_split["y_left"], depth + 1)
                right_node = self.build_tree(best_split["x_right"], best_split["y_right"], depth + 1)
                
                return Node(best_split["feature_index"], best_split["impurity"].round(3),
                            best_split["impurity_type"], best_split["samples"], left_node, right_node, best_split["value"], best_split["split_treshold"])
            else:
                print(X[best_split["feature_index"]])
                print(Y)
            
        print(Y.value_counts().reindex([0, 1], fill_value=0).tolist(), num_samples, depth)
        return Node(value=Y)
    
    def best_split(self, X, y):
        impurity = float('inf')
        new_impurity = 0
        best_x = []
        best_split = {}
        for feature_column in X.columns:
            x = X[feature_column]
            new_impurity = self.calculate_impurity(y, x)
            if new_impurity < impurity:
                impurity = new_impurity
                best_x = x
        
        best_split["feature_index"] = best_x.name
        best_split["impurity"] = impurity
        best_split["impurity_type"] = "Gini" if self.gini else "Entropy"
        best_split["x_left"] = X.loc[X[best_x.name] == 0].drop(columns=[best_x.name])
        best_split["y_left"] = y.loc[X[best_x.name] == 0]
        best_split["x_right"] = X.loc[X[best_x.name] == 1].drop(columns=[best_x.name])
        best_split["y_right"] = y.loc[X[best_x.name] == 1]
        best_split["value"] = y
        best_split["split_treshold"] = 0.5
        best_split["samples"] = num_samples = np.shape(X)[0]
        return best_split
    
    def calculate_probability(self, Feature):
        total = len(Feature)
        _, num_features = np.unique(Feature, return_counts=True)
        
        return num_features / total
    
    def calculate_conditional_probabilities(self, target, feature):
        total_elementos = len(target)
        unique_target = np.unique(target)
        unique_feature, counts_feature = np.unique(feature, return_counts=True)

        # Verificar si solo hay un tipo de número en feature
        if np.all(counts_feature == total_elementos) or len(unique_target) == 1:
            # Si solo hay un tipo de número, las probabilidades condicionales son uniformes
            probabilidades_condicionales = np.ones((len(unique_feature), len(unique_target))) / len(unique_target)
            return probabilidades_condicionales

        # Crear una matriz de índices para indexar target según los valores de feature
        indices = np.argsort(feature)
        feature_sorted = feature[indices]
        target_sorted = target[indices]

        # Usar np.add.reduceat para calcular los conteos condicionales
        counts_condicionales = np.add.reduceat(np.eye(len(unique_target))[target_sorted], np.searchsorted(feature_sorted, unique_feature))

        # Normalizar los conteos para obtener probabilidades condicionales
        probabilidades_condicionales = counts_condicionales / counts_feature[:, np.newaxis]

        return probabilidades_condicionales
    
    def calculate_impurity(self, target, feature):
        probabilidades_a = self.calculate_probability(feature.values)
        probabilidades_b = self.calculate_conditional_probabilities(target.values, feature.values)
        resultado = 0
        impureza = self.calculate_gini if self.gini else self.calculate_entropy
        
        for i in range(len(probabilidades_a)):
            resultado += probabilidades_a[i] * impureza(probabilidades_b[i][probabilidades_b[i]!=0])
        return resultado
    
    

    def calculate_gini(self, probabilidades):
        resultado = 0
        for i in probabilidades:
            resultado += (i **2)
        return 1 - resultado
    def calculate_entropy(self, probabilidades):
        resultado = 0
        resultado += probabilidades * np.log2(1/probabilidades)   
        
        return np.sum(resultado)
        
    def print_tree(self, node = None, indent=">"):   
        if not node:
            node = self.root
        
        if node.left is None and node.right is None:
            print(indent, node.value.value_counts().reindex([0, 1], fill_value=0).tolist())
        else:
            print(indent, "Feature: ",node.feature_index, node.impurity_type,": ",node.impurity,", ", node.value.value_counts().reindex([0, 1], fill_value=0).tolist())
            self.print_tree(node.right, "-" + indent)
            self.print_tree(node.left, "-" + indent)
    
    def visualize_tree(self, node = None, dot=None):
        if not node:
            node = self.root
        if dot is None:
            dot = Digraph()
            dot.node(name=str(id(node)), label=f"{node.feature_index}\n{node.impurity}\n{node.impurity_type}\n{node.samples}\n{node.value.value_counts().reindex([0, 1], fill_value=0).tolist()}")
        
        if node.left is not None:
            label=""
            if(node.left.feature_index is not None):
                label=f"{node.left.feature_index}\n{node.left.impurity}\n{node.left.impurity_type}\n{node.samples}\n"
            label +=f"{node.left.value.value_counts().reindex([0, 1], fill_value=0).tolist()}"
            dot.node(name=str(id(node.left)), label=label)
            dot.edge(str(id(node)), str(id(node.left)), label="Left")
            self.visualize_tree(node.left, dot)
        
        if node.right is not None:
            label=""
            if(node.right.feature_index is not None):
                label=f"{node.right.feature_index}\n{node.right.impurity}\n{node.right.impurity_type}\n{node.samples}\n"  
            label +=f"{node.right.value.value_counts().reindex([0, 1], fill_value=0).tolist()}"
            dot.node(name=str(id(node.right)), label=label)
            dot.edge(str(id(node)), str(id(node.right)), label="Right")
            self.visualize_tree(node.right, dot)
        
        return dot
        
    def predict(self, X):
        return X.apply(self.idividual_predict,  axis=1).values
        return
    
    def idividual_predict(self, x, node=None):
        if node is None:
                node = self.root

        if node.left is None and node.right is None:
            return node.value.mode().iloc[0]
        if x[node.feature_index] <= node.split_treshold:
            return self.idividual_predict(x, node.left)
        return self.idividual_predict(x, node.right)
        
#feature_index=None, impurity=None, impurity_type=None, left=None, right=None, value=None