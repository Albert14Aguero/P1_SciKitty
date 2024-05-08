import sys
sys.path.append("../")
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import joblib
import matplotlib.pyplot as plt
from models.DecisionTree import DecisionTree

#Tarea 9) Se repiten los pasos anteriores 6 y 8 usando el modelo salvado y se obtienen los mismos resultados.

df = pd.read_csv("../datasets/fictional_reading_place.csv")   


features_dictionary = {}
for column in df.columns:
    unique_categories = df[column].unique()
    features_dictionary[column] = list(unique_categories)

df['user_action'] = df['user_action'].replace({'skips':0, 'reads': 1})
df['author'] = df['author'].replace({'known':0, 'unknown': 1})
df['thread'] = df['thread'].replace({'new':0, 'follow_up': 1})
df['length'] = df['length'].replace({'long':0, 'short': 1}) 
df['where_read'] = df['where_read'].replace({'home':0, 'work': 1}) 

X = df.drop('user_action', axis=1).drop('example', axis=1)   #Tarea 3): Se carga el dataset en la forma usual a X,y
y = df['user_action']              #Tarea 3): Se carga el dataset en la forma usual a X,y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)      


trained_tree = joblib.load('../persist/fictional_reading_place.pkl')

predictions =  trained_tree.predict(X_test)



accuracy = Accuracy.accuracy_score(y_test, predictions)              #Tarea 6) : Se evalúa el árbol (valida usando X_test, y_test) mostrando exactitud, precisión, recall, F1. En los casos de target binario se muestra la matriz de confusión.
f1= F1.f1_score(y_test, predictions)                                         #Tarea 6) : Se evalúa el árbol (valida usando X_test, y_test) mostrando exactitud, precisión, recall, F1. En los casos de target binario se muestra la matriz de confusión.
recall = Recall.recall(y_test, predictions)                             #Tarea 6) : Se evalúa el árbol (valida usando X_test, y_test) mostrando exactitud, precisión, recall, F1. En los casos de target binario se muestra la matriz de confusión.
precision = Precision.precision(y_test, predictions)                             #Tarea 6) : Se evalúa el árbol (valida usando X_test, y_test) mostrando exactitud, precisión, recall, F1. En los casos de target binario se muestra la matriz de confusión.
conf_matrix = Confusion_matrix.confusion_matrix(y_test, predictions)                             #Tarea 6) : Se evalúa el árbol (valida usando X_test, y_test) mostrando exactitud, precisión, recall, F1. En los casos de target binario se muestra la matriz de confusión.
fpr = FPR.fpr(y_test, predictions)


print("SciKitty Metrics:")
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

trained_tree.print_tree()
dot = trained_tree.visualize_tree(features_dictionary=features_dictionary)
dot.render("../view/fictional_reading_place_tree", format="pdf", cleanup=True)
#plt.figure(figsize=(10, 6))                                                                             #Tarea 8) Se visualiza el árbol entrenado (puede ser generando un pdf).
#plot_tree(trained_tree, feature_names=X_train.columns, class_names=['No', 'Yes'], filled=True)           #Tarea 8) Se visualiza el árbol entrenado (puede ser generando un pdf).
#plt.show()                                                                                               #Tarea 8) Se visualiza el árbol entrenado (puede ser generando un pdf).


cls_sklearn = DecisionTreeClassifier(criterion='gini')                                               #Tarea 10) Opcional: se muestra en cada caso el árbol que saldría usando las librerías de sckitlearn

cls_sklearn.fit(X_train, y_train)                                                                       #Tarea 10) Opcional: se muestra en cada caso el árbol que saldría usando las librerías de sckitlearn

plt.figure(figsize=(10, 6))                                                                             #Tarea 10) Opcional: se muestra en cada caso el árbol que saldría usando las librerías de sckitlearn
plot_tree(cls_sklearn, feature_names=X_train.columns, class_names=['No', 'Yes'], filled=True)           #Tarea 10) Opcional: se muestra en cada caso el árbol que saldría usando las librerías de sckitlearn
plt.show()   