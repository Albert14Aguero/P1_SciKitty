import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from graphviz import Digraph
import json
import requests

class Node:
    def __init__(self, feature_index=None, impurity=None, impurity_type=None, samples=None, left=None, right=None, value=None, split_treshold=None, y_impurity=None):
        self.feature_index = feature_index
        self.impurity = impurity
        self.impurity_type = impurity_type
        self.samples = samples
        self.left = left
        self.right = right
        self.value = value
        self.split_treshold = split_treshold
        self.y_impurity = y_impurity


class DecisionTree:
    def __init__(self, min_samples = 1, max_depth = 4, gini = False):
        self.root = None
        self.min_samples = min_samples
        self.max_depth = max_depth
        self.gini = gini
    
    def fit(self, X, Y):
        self.root = self.build_tree(X, Y)
    
    def build_tree(self, X, Y, depth=0):
        num_samples, num_features = np.shape(X)
        if num_samples >= self.min_samples and depth <= self.max_depth and num_features > 0:
            
            best_split = self.best_split(X, Y)
            if best_split["impurity_type"] == "Entropy":
                y_impurity = self.calculate_y_entropy(best_split["value"])
            else:
                y_impurity = self.calculate_y_gini(best_split["value"])
                
            if y_impurity > 0:
                left_node = self.build_tree(best_split["x_left"], best_split["y_left"], depth + 1)
                right_node = self.build_tree(best_split["x_right"], best_split["y_right"], depth + 1)
                
                
                
                return Node(best_split["feature_index"], best_split["impurity"].round(3),
                            best_split["impurity_type"], best_split["samples"], left_node, right_node, best_split["value"], best_split["split_treshold"], y_impurity)
            
                
            
       
        return Node(value=Y)

    
    def calculate_y_entropy(self, target):
        def log2(p):
            return 0 if p == 0 else np.log2(p)
        log2 = np.vectorize(log2)
        p = self.calculate_probability(target.values)
        return -np.sum(p*log2(p))
    
    def calculate_y_gini(self, target):
        p = self.calculate_probability(target.values)
        return 1 - np.sum(p ** 2)

        
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
        total_elements = len(target)
        unique_target = np.unique(target)
        unique_feature, counts_feature = np.unique(feature, return_counts=True)

        if np.all(counts_feature == total_elements) or len(unique_target) == 1:
            # Si solo hay un tipo de número, las probabilidades condicionales son uniformes
            conditional_probabilities = np.ones((len(unique_feature), len(unique_target))) / len(unique_target)
            return conditional_probabilities

        # Crear una matriz de índices para indexar target según los valores de feature
        indices = np.argsort(feature)
        feature_sorted = feature[indices]
        target_sorted = target[indices]

        # Usar np.add.reduceat para calcular los conteos condicionales
        counts_conditional = np.add.reduceat(np.eye(len(unique_target), dtype=int)[target_sorted.astype(int)], np.searchsorted(feature_sorted, unique_feature))

        # Normalizar los conteos para obtener probabilidades condicionales
        conditional_probabilities = counts_conditional / counts_feature[:, np.newaxis]

        return conditional_probabilities


    
    def calculate_impurity(self, target, feature):
        probabilities_a = self.calculate_probability(feature.values)
        probabilities_b = self.calculate_conditional_probabilities(target.values, feature.values)
        result = 0
        impurity_func = self.calculate_gini if self.gini else self.calculate_entropy
        
        for i in range(len(probabilities_a)):
            non_zero_indices = probabilities_b[i] != 0
            result += probabilities_a[i] * impurity_func(probabilities_b[i][non_zero_indices])
        return result

    
    

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
    
    def visualize_tree(self, node=None, dot=None, features_dictionary = None):
        
        if not node:
            node = self.root
        if dot is None:
            dot = Digraph()
            dot.node(name=str(id(node)), label=f"{node.feature_index}\n{node.impurity_type} = {node.y_impurity}\nValues = {node.value.value_counts().reindex([0, 1], fill_value=0).tolist()}\nSamples = {node.samples}")
        
        if node.left is not None:
            label = ""
            if node.left.feature_index is not None:
                label = f"{node.left.feature_index}\n{node.impurity_type} = {node.left.y_impurity}\nSamples = {node.left.samples}\n"
            label += f"Values = {node.left.value.value_counts().reindex([0, 1], fill_value=0).tolist()}"
            dot.node(name=str(id(node.left)), label=label)
            right_label = "0" if features_dictionary == None else features_dictionary[node.feature_index][0]
            dot.edge(str(id(node)), str(id(node.left)), label=right_label)
            self.visualize_tree(node.left, dot, features_dictionary)
        
        if node.right is not None:
            label = ""
            if node.right.feature_index is not None:
                label = f"{node.right.feature_index}\n{node.impurity_type} = {node.right.y_impurity}\nSamples = {node.right.samples}\n"
            label += f"Values = {node.right.value.value_counts().reindex([0, 1], fill_value=0).tolist()}"
            dot.node(name=str(id(node.right)), label=label)
            left_label = "1" if features_dictionary == None else features_dictionary[node.feature_index][1]
            dot.edge(str(id(node)), str(id(node.right)), label=left_label)
            self.visualize_tree(node.right, dot, features_dictionary)
        
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
    
    def to_dict(self, node=None):  # Añadido y_feature_name como parámetro
        if node is None:
            node = self.root

        node_dict = {
            'feature_index': node.feature_index,
            'impurity': node.impurity,
            'impurity_type': node.impurity_type,
            'samples': node.samples,
            'value': node.value.value_counts().reindex([0, 1], fill_value=0).tolist(),
            'split_treshold': node.split_treshold,
            'y_impurity': node.y_impurity,
        }
        if node.left is not None:
            node_dict['left'] = self.to_dict(node.left)
        if node.right is not None:
            node_dict['right'] = self.to_dict(node.right)
        
        return node_dict

    def to_json(self, y_feature_name, y_feature_values=None):
        tree_dict = self.to_dict()
        result = {
            'tree': tree_dict,
            'y_feature': {
                'name': y_feature_name,
                'values': y_feature_values
            }
        }
        return json.dumps(result)
    
    def export_to_prolog(self, url, y_feature_name, y_feature_values=None):
        headers = {'Content-Type': 'application/json'}
        response = requests.post(url, data=self.to_json(y_feature_name, y_feature_values), headers=headers)
        
        if response.status_code == 200:
            print("Successfully sent the JSON to the specified URL.")
        else:
            print(f"Failed to send JSON. Status code: {response.status_code}, Response: {response.text}")

    def prediction_by_prolog(self, url, x_predict):
        print(json.dumps(x_predict))
        headers = {'Content-Type': 'application/json'}
        response = requests.post(url, data=json.dumps(x_predict), headers=headers)
        
        if response.status_code == 200:
            print(response.json())
            return response.json()
        else:
            print(f"Failed to send JSON. Status code: {response.status_code}, Response: {response.text}")
#feature_index=None, impurity=None, impurity_type=None, left=None, right=None, value=None
