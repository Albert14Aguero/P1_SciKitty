import sys
sys.path.append("../")
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from metrics import Accuracy, F1, Recall, FPR, Precision, Confusion_matrix

df = pd.read_csv("../datasets/playTennis.csv")   #Tarea 3): Se carga el dataset en la forma usual a X,y

df_encoded = pd.get_dummies(df, columns=['Outlook', 'Temperature', 'Humidity', 'Wind'],dtype = "int") #Tarea 3): Se carga el dataset en la forma usual a X,y

df_encoded['Play Tennis'] = df_encoded['Play Tennis'].map({'No': 0, 'Yes': 1})  #Tarea 3): Se carga el dataset en la forma usual a X,y

features_dictionary = {}
for column in df_encoded.columns:
    unique_categories = df_encoded[column].unique()
    numbers = sorted(list(unique_categories))
    features_dictionary[column] =  numbers if  numbers != [0,1] else [{0: "No", 1: "Yes"}[i] for i in numbers]

X = df_encoded.drop('Play Tennis', axis=1)  #Tarea 3): Se carga el dataset en la forma usual a X,y
y = df_encoded['Play Tennis']               #Tarea 3): Se carga el dataset en la forma usual a X,y

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)      #Tarea 4): Se crea crean X_train, y_train, X_test, y_test


trained_tree = joblib.load('../persist/play_tennis.pkl')

predictions =  trained_tree.predict(X_test)

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



dot = trained_tree.visualize_tree(features_dictionary=features_dictionary)                                   #Tarea 8) Se visualiza el árbol entrenado (puede ser generando un pdf).
dot.render("../view/play_tennis_tree", format="pdf", cleanup=True)
print("Graph generated as play_tennis_tree.pdf")


cls_sklearn = DecisionTreeClassifier(criterion='entropy')                                               #Tarea 10) Opcional: se muestra en cada caso el árbol que saldría usando las librerías de sckitlearn

cls_sklearn.fit(X_train, y_train)                                                                       #Tarea 10) Opcional: se muestra en cada caso el árbol que saldría usando las librerías de sckitlearn

plt.figure(figsize=(10, 6))                                                                             #Tarea 10) Opcional: se muestra en cada caso el árbol que saldría usando las librerías de sckitlearn
plot_tree(cls_sklearn, feature_names=X_train.columns, class_names=['No', 'Yes'], filled=True)           #Tarea 10) Opcional: se muestra en cada caso el árbol que saldría usando las librerías de sckitlearn
plt.show()   