import sys
sys.path.append("../models")
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import joblib
import matplotlib.pyplot as plt
from DecisionTree import DecisionTree

#Tarea 9) Se repiten los pasos anteriores 6 y 8 usando el modelo salvado y se obtienen los mismos resultados.

df = pd.read_csv("../datasets/fictional_reading_place.csv")   

df['user_action'] = df['user_action'].replace({'skips':0, 'reads': 1})
df['author'] = df['author'].replace({'known':0, 'unknown': 1})
df['thread'] = df['thread'].replace({'new':0, 'follow_up': 1})
df['length'] = df['length'].replace({'long':0, 'short': 1}) 
df['where_read'] = df['where_read'].replace({'home':0, 'work': 1}) 

X = df.drop('user_action', axis=1).drop('example', axis=1)   #Tarea 3): Se carga el dataset en la forma usual a X,y
y = df['user_action']              #Tarea 3): Se carga el dataset en la forma usual a X,y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)      


trained_tree = joblib.load('../persist/fictional_reading_place.pkl')

preds =  trained_tree.predict(X_test)

trained_tree.print_tree()
dot = trained_tree.visualize_tree()
dot.render("fictional_reading_place", format="pdf", cleanup=True)
#plt.figure(figsize=(10, 6))                                                                             #Tarea 8) Se visualiza el árbol entrenado (puede ser generando un pdf).
#plot_tree(trained_tree, feature_names=X_train.columns, class_names=['No', 'Yes'], filled=True)           #Tarea 8) Se visualiza el árbol entrenado (puede ser generando un pdf).
#plt.show()                                                                                               #Tarea 8) Se visualiza el árbol entrenado (puede ser generando un pdf).



cls_sklearn = DecisionTreeClassifier(criterion='gini')                                               #Tarea 10) Opcional: se muestra en cada caso el árbol que saldría usando las librerías de sckitlearn

cls_sklearn.fit(X_train, y_train)                                                                       #Tarea 10) Opcional: se muestra en cada caso el árbol que saldría usando las librerías de sckitlearn

plt.figure(figsize=(10, 6))                                                                             #Tarea 10) Opcional: se muestra en cada caso el árbol que saldría usando las librerías de sckitlearn
plot_tree(cls_sklearn, feature_names=X_train.columns, class_names=['No', 'Yes'], filled=True)           #Tarea 10) Opcional: se muestra en cada caso el árbol que saldría usando las librerías de sckitlearn
plt.show()   