import sys
sys.path.append("../")
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from models.DecisionTree import DecisionTree
from metrics import Accuracy, F1, Recall, FPR, Precision, Confusion_matrixe

df = pd.read_csv("../datasets/fictional_reading_place.csv")   #Tarea 3): Se carga el dataset en la forma usual a X,y

df['user_action'] = df['user_action'].replace({'skips':0, 'reads': 1})
df['author'] = df['author'].replace({'known':0, 'unknown': 1})
df['thread'] = df['thread'].replace({'new':0, 'follow_up': 1})
df['length'] = df['length'].replace({'long':0, 'short': 1}) 
df['where_read'] = df['where_read'].replace({'home':0, 'work': 1}) 

X = df.drop('user_action', axis=1).drop('example', axis=1)   #Tarea 3): Se carga el dataset en la forma usual a X,y
y = df['user_action']              #Tarea 3): Se carga el dataset en la forma usual a X,y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)      #Tarea 4): Se crea crean X_train, y_train, X_test, y_test

print(X_train)
print(y_train)
cls = DecisionTree(gini = False)  #Tarea 5): Se entrena con X_train (método fit) el árbol.

cls.fit(X_train, y_train)    #Tarea 5): Se entrena con X_train (método fit) el árbol.    

cls.print_tree()

predictions =  cls.predict(X_test)  

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
                                                                            #Tarea 6) : Se evalúa el árbol (valida usando X_test, y_test) mostrando exactitud, precisión, recall, F1. En los casos de target binario se muestra la matriz de confusión.


joblib.dump(cls, '../persist/fictional_reading_place.pkl')                                 #Tarea 7): Se salva (exporta, serializa) el modelo.


dot = cls.visualize_tree()                                   #Tarea 8) Se visualiza el árbol entrenado (puede ser generando un pdf).
dot.render("../view/fictional_reading_place", format="pdf", cleanup=True)
print("Graph generated as fictional_reading_place.pdf")
