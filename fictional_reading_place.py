import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scikitty.models.DecisionTree import DecisionTree

df = pd.read_csv("scikitty/datasets/fictional_reading_place.csv")   #Tarea 3): Se carga el dataset en la forma usual a X,y

df['user_action'] = df['user_action'].replace({'skips':0, 'reads': 1})
df['author'] = df['author'].replace({'unknown':0, 'known': 1})
df['thread'] = df['thread'].replace({'new':0, 'follow_up': 1})
df['length'] = df['length'].replace({'long':0, 'short': 1})

df_encoded = pd.get_dummies(df, columns=['user_action', 'author', 'thread', 'length'],drop_first=True,dtype = "int") #Tarea 3): Se carga el dataset en la forma usual a X,y

df_encoded['where_read'] = df_encoded['where_read'].map({'home': 0, 'work': 1})  #Tarea 3): Se carga el dataset en la forma usual a X,y

X = df_encoded.drop('where_read', axis=1).drop('example', axis=1)  #Tarea 3): Se carga el dataset en la forma usual a X,y
y = df_encoded['where_read']               #Tarea 3): Se carga el dataset en la forma usual a X,y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)      #Tarea 4): Se crea crean X_train, y_train, X_test, y_test


cls = DecisionTree(gini = True)  #Tarea 5): Se entrena con X_train (método fit) el árbol.

cls.fit(X_train, y_train)    #Tarea 5): Se entrena con X_train (método fit) el árbol.    

cls.print_tree()
dot = cls.visualize_tree()
dot.render("tree", format="pdf", cleanup=True)
print("Graph generated as tree.pdf")

preds =  cls.predict(X_test)  #Tarea 6) : Se evalúa el árbol (valida usando X_test, y_test) mostrando exactitud, precisión, recall, F1. En los casos de target binario se muestra la matriz de confusión.
print(preds)
print(y_test)
"""


joblib.dump(cls, 'scikitty/persist/playTennis.pkl')                                 #Tarea 7): Se salva (exporta, serializa) el modelo.

#plt.figure(figsize=(10, 6))                                                                             #Tarea 8) Se visualiza el árbol entrenado (puede ser generando un pdf).
#plot_tree(trained_tree, feature_names=X_train.columns, class_names=['No', 'Yes'], filled=True)           #Tarea 8) Se visualiza el árbol entrenado (puede ser generando un pdf).
#plt.show()                                                                                               #Tarea 8) Se visualiza el árbol entrenado (puede ser generando un pdf).
"""