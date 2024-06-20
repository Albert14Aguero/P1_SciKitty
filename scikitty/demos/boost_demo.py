import sys
sys.path.append("../")
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from models.DecisionTree import DecisionTree
from models.TreeGradientBoosting import tree_gradient_boosting
from metrics import Accuracy, F1, Recall, FPR, Precision, Confusion_matrix
import matplotlib.pyplot as plt

# Cargar el conjunto de datos
df = pd.read_csv("../datasets/fake_weights.csv",  sep=';')   #Tarea 3): Se carga el dataset en la forma usual a X,y

mean_weight = df['Weight(y)'].mean()
mean_height = df['Height'].mean()

# Crear una columna binaria basada en el promedio de Weight(y)
binary_values_w = (df['Weight(y)'] >= mean_weight).astype(int)
binary_values_h = (df['Height'] >= mean_height).astype(int)

# Reemplazar la columna Weight(y) por la columna binaria
df['Weight(y)'] = binary_values_w 
df['Height'] = binary_values_h 

df['Gender'] = df['Gender'].map({'male': 0, 'female': 1})

# Codificar la variable categórica 'Color' usando one-hot encoding
df_encoded = pd.get_dummies(df, columns=['Color'],dtype = "int")

# Separar características (X) y objetivo (y)
X = df_encoded.drop(['id', 'Gender'], axis=1)
y = df_encoded['Gender']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


gbc = tree_gradient_boosting(alpha=0.1, T=100, criterion='gini')

gbc.fit(X_train, y_train)

predictions_test = gbc.predict(X_test)

cls = DecisionTree(gini = True)  #Tarea 5): Se entrena con X_train (método fit) el árbol.

cls.fit(X_train, y_train)    #Tarea 5): Se entrena con X_train (método fit) el árbol.  

predicciones = cls.predict(X_test)

print("Decision Tree Prediction = ", predicciones)
# Mostrar resultados (en un ejemplo real se evaluaría con datos de prueba)
print("Tree Gradient Boosting Prediction:", predictions_test)
print("Ground Truth = ", [y_test[0], y_test[1]])

# Evaluar el rendimiento (ejemplo con exactitud para problemas de clasificación)
