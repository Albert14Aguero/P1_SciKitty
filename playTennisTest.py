import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import joblib
import matplotlib.pyplot as plt

#Tarea 9) Se repiten los pasos anteriores 6 y 8 usando el modelo salvado y se obtienen los mismos resultados.

df = pd.read_csv("scikitty/datasets/playTennis.csv")   

df_encoded = pd.get_dummies(df, columns=['Outlook', 'Temperature', 'Humidity', 'Wind'],dtype = "int") 

df_encoded['Play Tennis'] = df_encoded['Play Tennis'].map({'No': 0, 'Yes': 1})  

X = df_encoded.drop('Play Tennis', axis=1)  
y = df_encoded['Play Tennis']               

X_train , X_test, y_train , y_test = train_test_split(X, y, test_size=0.2, random_state=42)      


trained_tree = joblib.load('scikitty/persist/playTennis.pkl')

preds =  trained_tree.predict(X_test)

#plt.figure(figsize=(10, 6))                                                                             #Tarea 8) Se visualiza el árbol entrenado (puede ser generando un pdf).
#plot_tree(trained_tree, feature_names=X_train.columns, class_names=['No', 'Yes'], filled=True)           #Tarea 8) Se visualiza el árbol entrenado (puede ser generando un pdf).
#plt.show()                                                                                               #Tarea 8) Se visualiza el árbol entrenado (puede ser generando un pdf).



cls_sklearn = DecisionTreeClassifier(criterion='entropy')                                               #Tarea 10) Opcional: se muestra en cada caso el árbol que saldría usando las librerías de sckitlearn

cls_sklearn.fit(X_train, y_train)                                                                       #Tarea 10) Opcional: se muestra en cada caso el árbol que saldría usando las librerías de sckitlearn

plt.figure(figsize=(10, 6))                                                                             #Tarea 10) Opcional: se muestra en cada caso el árbol que saldría usando las librerías de sckitlearn
plot_tree(cls_sklearn, feature_names=X_train.columns, class_names=['No', 'Yes'], filled=True)           #Tarea 10) Opcional: se muestra en cada caso el árbol que saldría usando las librerías de sckitlearn
plt.show()   