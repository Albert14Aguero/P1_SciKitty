import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from ..models.LogisticRegression import LogisticRegression

df = pd.read_csv("../datasets/playTennis.csv")   #Tarea 3): Se carga el dataset en la forma usual a X,y

df_encoded = pd.get_dummies(df, columns=['Outlook', 'Temperature', 'Humidity', 'Wind'],dtype = "int") #Tarea 3): Se carga el dataset en la forma usual a X,y

df_encoded['Play Tennis'] = df_encoded['Play Tennis'].map({'No': 0, 'Yes': 1})  #Tarea 3): Se carga el dataset en la forma usual a X,y

X = df_encoded.drop('Play Tennis', axis=1)  #Tarea 3): Se carga el dataset en la forma usual a X,y
y = df_encoded['Play Tennis']               #Tarea 3): Se carga el dataset en la forma usual a X,y

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)      #Tarea 4): Se crea crean X_train, y_train, X_test, y_test

cls = LogisticRegression()  #Tarea 5): Se entrena con X_train (método fit) el árbol.

cls.fit(X_train, y_train)    #Tarea 5): Se entrena con X_train (método fit) el árbol.    

preds =  cls.predict(X_test)  #Tarea 6) : Se evalúa el árbol (valida usando X_test, y_test) mostrando exactitud, precisión, recall, F1. En los casos de target binario se muestra la matriz de confusión.


print(preds)

joblib.dump(cls, '../persist/playTennis.pkl')                                 #Tarea 7): Se salva (exporta, serializa) el modelo.

#plt.figure(figsize=(10, 6))                                                                             #Tarea 8) Se visualiza el árbol entrenado (puede ser generando un pdf).
#plot_tree(trained_tree, feature_names=X_train.columns, class_names=['No', 'Yes'], filled=True)           #Tarea 8) Se visualiza el árbol entrenado (puede ser generando un pdf).
#plt.show()                                                                                               #Tarea 8) Se visualiza el árbol entrenado (puede ser generando un pdf).