import sys
sys.path.append("../")
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from metrics import Accuracy, F1, Recall, FPR, Precision, Confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

df = pd.read_csv("../datasets/fictional_disease.csv")   

features_dictionary = {}
for column in df.columns:
    unique_categories = df[column].unique()
    features_dictionary[column] = list(unique_categories)
features_dictionary['Age'] = ['Younger', 'Older']

meanAge = df['Age'].mean() #Obtenemos el priomedio de la edad
binary_values = (df['Age'] <= meanAge) #Obtenemos un arreglo de booleanos que nos dice si la edad es menor o igual al promedio
df['Age'] = binary_values  #Reemplazamos la columna de edad por el arreglo de booleanos
df['Age'] = df['Age'].replace({False:0, True: 1}) #Reemplazamos el arreglo de booleanos por 0 y 1
df['Gender'] = df['Gender'].replace({'Male':0, 'Female': 1})
df['SmokerHistory'] = df['SmokerHistory'].replace({'Non_smoker':0, 'Smoker': 1})
df['Disease'] = df['Disease'].map({'not_diseased': 0, 'diseased': 1}) 


y = df['Disease']
X = df.drop('Disease', axis=1)
                           #Tarea 3): Se carga el dataset en la forma usual a X,y

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)      #Tarea 4): Se crea crean X_train, y_train, X_test, y_test


trained_tree = joblib.load('../persist/fictional_disease.pkl')

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


accuracy_sklearn = accuracy_score(y_test, predictions)
precision_sklearn = precision_score(y_test, predictions)
recall_sklearn = recall_score(y_test, predictions)
f1_sklearn = f1_score(y_test, predictions)
conf_matrix_sklearn = confusion_matrix(y_test, predictions)

# Imprimir las métricas de sklearn
print("\n\nSklearn Metrics:")
print("Accuracy =", accuracy_sklearn)
print("Precision =", precision_sklearn)
print("Recall =", recall_sklearn)
print("F1 =", f1_sklearn)
print("Confusion Matrix =\n", conf_matrix_sklearn)

dot = trained_tree.visualize_tree(features_dictionary=features_dictionary)                                   #Tarea 8) Se visualiza el árbol entrenado (puede ser generando un pdf).
dot.render("../view/ficitional_disease_tree", format="pdf", cleanup=True)
print("Graph generated as play_tennis_tree.pdf")


cls_sklearn = DecisionTreeClassifier(criterion='entropy')                                               #Tarea 10) Opcional: se muestra en cada caso el árbol que saldría usando las librerías de sckitlearn

cls_sklearn.fit(X_train, y_train)                                                                       #Tarea 10) Opcional: se muestra en cada caso el árbol que saldría usando las librerías de sckitlearn

plt.figure(figsize=(10, 6))                                                                             #Tarea 10) Opcional: se muestra en cada caso el árbol que saldría usando las librerías de sckitlearn
plot_tree(cls_sklearn, feature_names=X_train.columns, class_names=['No', 'Yes'], filled=True)           #Tarea 10) Opcional: se muestra en cada caso el árbol que saldría usando las librerías de sckitlearn
plt.show()   