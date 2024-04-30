import sys
sys.path.append("../")
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from models.DecisionTree import DecisionTree
from metrics import Accuracy, F1, Recall, FPR, Precision, Confusion_matrix

#Tarea 9) Se repiten los pasos anteriores 6 y 8 usando el modelo salvado y se obtienen los mismos resultados.

df = pd.read_csv("../datasets/fictional_disease.csv")   



meanAge = df['Age'].mean() #Obtenemos el priomedio de la edad
binary_values = (df['Age'] <= meanAge) #Obtenemos un arreglo de booleanos que nos dice si la edad es menor o igual al promedio
df['Age'] = binary_values  #Reemplazamos la columna de edad por el arreglo de booleanos
df['Age'] = df['Age'].replace({False:0, True: 1}) #Reemplazamos el arreglo de booleanos por 0 y 1
df['Gender'] = df['Gender'].replace({'Male':0, 'Female': 1})
df['SmokerHistory'] = df['SmokerHistory'].replace({'Non_smoker':0, 'Smoker': 1})
df['Disease'] = df['Disease'].map({'not_diseased': 0, 'diseased': 1}) 


y = df['Disease']
X = df.drop('Disease', axis=1)
               

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)      

cls = DecisionTree(gini = False)  #Tarea 5): Se entrena con X_train (método fit) el árbol.

cls.fit(X_train, y_train)    #Tarea 5): Se entrena con X_train (método fit) el árbol.    

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

joblib.dump(cls, '../persist/fictional_disease.pkl')                                 #Tarea 7): Se salva (exporta, serializa) el modelo.

dot = cls.visualize_tree()                                               #Tarea 8) Se visualiza el árbol entrenado (puede ser generando un pdf).
dot.render("../view/fictional_disease", format="pdf", cleanup=True)
print("Graph generated as fictional_disease.pdf")