import sys
sys.path.append("../")
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from models.DecisionTree import DecisionTree
from metrics import Accuracy, F1, Recall, FPR, Precision, Confusion_matrix

df = pd.read_csv("../datasets/playTennis.csv")   #Tarea 3): Se carga el dataset en la forma usual a X,y

df_encoded = pd.get_dummies(df, columns=['Outlook', 'Temperature', 'Humidity', 'Wind'],dtype = "int") #Tarea 3): Se carga el dataset en la forma usual a X,y

df_encoded['Play Tennis'] = df_encoded['Play Tennis'].map({'No': 0, 'Yes': 1})  #Tarea 3): Se carga el dataset en la forma usual a X,y

X = df_encoded.drop('Play Tennis', axis=1)  #Tarea 3): Se carga el dataset en la forma usual a X,y
y = df_encoded['Play Tennis']               #Tarea 3): Se carga el dataset en la forma usual a X,y

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)      #Tarea 4): Se crea crean X_train, y_train, X_test, y_test

cls = DecisionTree(gini = False)  #Tarea 5): Se entrena con X_train (método fit) el árbol.

cls.fit(X_train, y_train)    #Tarea 5): Se entrena con X_train (método fit) el árbol.    


#cls.print_tree()

predictions =  cls.predict(X_test)  #Tarea 6) : Se evalúa el árbol (valida usando X_test, y_test) mostrando exactitud, precisión, recall, F1. En los casos de target binario se muestra la matriz de confusión.

accuracy = Accuracy.accuracy_score(y_test, predictions)              #Tarea 6) : Se evalúa el árbol (valida usando X_test, y_test) mostrando exactitud, precisión, recall, F1. En los casos de target binario se muestra la matriz de confusión.
f1= F1.f1_score(y_test, predictions)                                         #Tarea 6) : Se evalúa el árbol (valida usando X_test, y_test) mostrando exactitud, precisión, recall, F1. En los casos de target binario se muestra la matriz de confusión.
recall = Recall.recall(y_test, predictions)                             #Tarea 6) : Se evalúa el árbol (valida usando X_test, y_test) mostrando exactitud, precisión, recall, F1. En los casos de target binario se muestra la matriz de confusión.
precision = Precision.precision(y_test, predictions)                             #Tarea 6) : Se evalúa el árbol (valida usando X_test, y_test) mostrando exactitud, precisión, recall, F1. En los casos de target binario se muestra la matriz de confusión.
conf_matrix = Confusion_matrix.confusion_matrix(y_test, predictions)                             #Tarea 6) : Se evalúa el árbol (valida usando X_test, y_test) mostrando exactitud, precisión, recall, F1. En los casos de target binario se muestra la matriz de confusión.
fpr = FPR.fpr(y_test, predictions)



print("Accuracy =", accuracy)                                        #Tarea 6) : Se evalúa el árbol (valida usando X_test, y_test) mostrando exactitud, precisión, recall, F1. En los casos de target binario se muestra la matriz de confusión.
print("F1 =", f1)                                                     #Tarea 6) : Se evalúa el árbol (valida usando X_test, y_test) mostrando exactitud, precisión, recall, F1. En los casos de target binario se muestra la matriz de confusión.
print("Recall =", recall)                                                     #Tarea 6) : Se evalúa el árbol (valida usando X_test, y_test) mostrando exactitud, precisión, recall, F1. En los casos de target binario se muestra la matriz de confusión.
print("Precision =", precision)                                                     #Tarea 6) : Se evalúa el árbol (valida usando X_test, y_test) mostrando exactitud, precisión, recall, F1. En los casos de target binario se muestra la matriz de confusión.
print("Confusion Matrix =\n", conf_matrix)                                                     #Tarea 6) : Se evalúa el árbol (valida usando X_test, y_test) mostrando exactitud, precisión, recall, F1. En los casos de target binario se muestra la matriz de confusión.

print("FPR =", fpr)                                                     #Tarea 6) : Se evalúa el árbol (valida usando X_test, y_test) mostrando exactitud, precisión, recall, F1. En los casos de target binario se muestra la matriz de confusión.

joblib.dump(cls, '../persist/play_tennis.pkl')                                 #Tarea 7): Se salva (exporta, serializa) el modelo.

dot = cls.visualize_tree()                                               #Tarea 8) Se visualiza el árbol entrenado (puede ser generando un pdf).
dot.render("../view/play_tennis_tree", format="pdf", cleanup=True)
print("Graph generated as play_tennis_tree.pdf")