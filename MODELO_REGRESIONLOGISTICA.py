import string
import nltk
import numpy as np
import re
import csv
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import KFold
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
import argparse
import csv
import datetime
from sklearn import metrics
import sklearn.metrics as metrics
from tabulate import tabulate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

#FUNCION PARA CARGAR DATOS DE ENTRENAMIENTO 
def load_df():
    file_name = 'D:/CURSOS/CICLO REGULAR-2021-1/SEM. INV/Intento/data_excel2_2.0__.xlsx'
    train  = pd.read_excel(file_name)
    return train 

#FUNCION PARA CARGAR DATOS DE PRUEBA 
def load_df_pred():
    file_name = 'D:/CURSOS/CICLO REGULAR-2021-1/SEM. INV/Intento/data_excel_test2.xlsx'
    test  = pd.read_excel(file_name)
    return test 


#CARGAR DATOS DE ENTRENAMIENTO Y DE PRUEBA 

train = load_df()
test=load_df_pred()
print (train.shape)
print (test.shape)

#STOPWORDS Y LEMATIZACION 
list_stop_words = set(stopwords.words('english'))
wn=nltk.WordNetLemmatizer()

#FUNCION DE LIMPIEZA DE TEXTO 
def limpieza (texto):
    texto = texto.lower()
    texto = re.sub('((www\.[^\s]+)|(https?://[^\s]+))|(http?://[^\s]+)', '', texto)
    texto = re.sub('[\s]+', ' ', texto)
    texto = re.sub(r'#([^\s]+)', r'\1', texto)
    texto = re.sub(r'@([^\s]+)', '', texto) 
    texto = re.sub('&[a-z]+;', '', texto)
    texto = re.sub(r'\d+', '', texto)
    texto = texto.translate(str.maketrans(string.punctuation,' '*len(string.punctuation)))
    texto = texto.replace("'", "")
    texto = " ".join([word for word in texto.split(" ") if word not in list_stop_words])
    texto = texto.strip()
    return texto

#FUNCION DE LEMATIZACION 
def lematizacion(texto):
    texto=[wn.lemmatize(word) for word in texto]
    texto= ''.join(texto)
    #print (texto)
    #print ("***********************")
    return texto

#LIMPIAR Y APLICAR LA LEMATIZACION EN LA COLUMNA DE TWEETS DE ENTRENAMIENTO Y DE PRUEBA 
train['tweets'] = train['tweets'].apply(limpieza)
train['tweets'] = train['tweets'].apply(lematizacion)
#print (train['tweets'] )
test['tweets'] = test['tweets'].apply(limpieza)
test['tweets'] = test['tweets'].apply(lematizacion)
#print (test['tweets'])

#COLUMNAS DE ETIQUETAS 
train_label = train['y']
print ("#####################################")
print (train_label)
print ("#####################################")
test_label = test['y']

#APLICAR TF-IDF EN LA COLUMNA DE LOS WEETS DE ENTRENAMIENTO Y DE PRUEBA 
vectorizer = TfidfVectorizer(stop_words='english', max_features = 3000)
X_train=train['tweets'] 
X_test=test['tweets']
tfidf_train = vectorizer.fit_transform(X_train).toarray()
tfidf_test = vectorizer.transform(X_test).toarray()
print(X_train)
#print (tfidf_train)
print(vectorizer.get_feature_names())
print(vectorizer.vocabulary_)
#print ("//////////////////////////////////////////")

#MODELO DE REGRESION LOGISTICA
# función sigmoidea
def sigmoidea(X, weight):
    z = np.dot(X, weight)
    return 1 / (1 + np.exp(-z))

# calculando el gradiente DESCENDIENTE (estimar mejores coficientes)
def gradient_descent(X, h, y):
    return np.dot(X.T, (h - y)) / y.shape[0]

#actualización de los pesos
#que es el peso: tasa de aprendizaje multiplicada por gradiente
def peso_actualizado(weight, learning_rate, gradient):
    return weight - learning_rate * gradient

def predict(x, theta):
    theta_new = theta[:, np.newaxis]
    return sigmoidea(x,theta_new)

# Iterar y aprender los parámetros
def gradient(X, y):
    num_iter = 100
    theta = np.zeros(X.shape[1])
    for i in range(num_iter):
        h = sigmoidea(X, theta)
        gradient = gradient_descent(X, h, y)
        theta = peso_actualizado(theta, 0.1, gradient)
    return theta

# La probabilidad promedio de 0 y 1 en los datos de entrenamiento es de alrededor de 0.5
#por lo que establecemos el umbral para la clasificación en esta media
def accuracy_score(actual, pred):
    predicted_class = ((pred >= 0.5) .astype(int))
    predicted_class = predicted_class.flatten()
    acc = np.mean(predicted_class == actual)
    return acc

#VALIDACION CRUZADA
# Inicializar la validación cruzada de 10 veces
kfold = KFold(10, True, 1)
bestaccuracy = 0
scores = np.array([])
theta_final = np.zeros(tfidf_train.shape[1])

#Dividir los datos del tren para entrenar y validar e iterar sobre pliegues
for train, test in kfold.split(tfidf_train):
    X_train = tfidf_train[train]
    X_test = tfidf_train[test]    
    Y_train = train_label[train]
    #print (Y_train.shape)
    Y_test = train_label[test]
    #print (Y_test.shape)

    theta_out = gradient(X_train, Y_train)
    pred = predict(X_test, theta_out)
    acc_score = accuracy_score(Y_test, pred)
    #print (acc_score)

# Precisión para cada pliegue
    scores = np.append(scores, acc_score)
    
    if(acc_score > bestaccuracy):
        theta_final = theta_out
        bestaccuracy = acc_score
print("")
print("Los valores de accuracy en validaciones cruzadas de 10 veces son: ", scores)
print ("")
print("Promedio de accuracy: ", scores.mean())
print ("")

test_predicted = predict(tfidf_test, theta_final)

# Precisión en los datos de prueba
accuracy_test = accuracy_score(test_label, test_predicted)
print ("Acurracy en los datos de prueba: ")
print (accuracy_test)
print ("")
# Create confusion matrix
confusionMatrix = pd.DataFrame(data = confusion_matrix(test_label, (test_predicted >= 0.5) .astype(int)), 
                               columns=["0", "1"], index = ["0", "1"])
print ("Matriz de confusión")
print(confusionMatrix)
print ("")
# Precision = TP/(TP + FP)
precision = round((confusionMatrix.iloc[1, 1] / (confusionMatrix.iloc[1, 1] + confusionMatrix.iloc[0, 1])) * 100, 2)
print("Precision: ", precision)

# Recall = TP/(TP + FN)
recall = round((confusionMatrix.iloc[1, 1] / (confusionMatrix.iloc[1, 1] + confusionMatrix.iloc[1, 0])) * 100, 2)
print("Recall: ", recall)

# calculate roc curve
fpr, tpr, thresholds = metrics.roc_curve(test_label, test_predicted)
lr_auc = metrics.roc_auc_score(test_label, test_predicted)
print ("AUC")
print (lr_auc)

#F1
f1=2*((precision*recall)/(precision+recall))
print ("F1 Score: ", f1)

                                      

