from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import nltk
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import FitFailedWarning
import warnings

# IMPORTAR DATA

datasets = pd.read_excel('D:/CURSOS/CICLO REGULAR-2021-1/SEM. INV/Intento/data_excel2_2.0__.xlsx')
X= datasets['tweets']
Y=datasets['y']


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
X = X.apply(limpieza)
X = X.apply(lematizacion)
#print (X)

#ESCALA DE CARACTERISTICAS 
#APLICAR TF-IDF EN LA COLUMNA DE LOS WEETS DE ENTRENAMIENTO Y DE PRUEBA 
vectorizer = TfidfVectorizer(stop_words='english', max_features = 10000,
                             ngram_range=(2,2), analyzer='char_wb')
X = vectorizer.fit_transform(X).toarray()
#tfidf_test = vectorizer.fit_transform(X_Test).toarray()
#tfidf_test = vectorizer.transform(X_Test).toarray()
#print(X_Train)
#print (X)


# DIVIDIR EL CONJUNTO DE DATOS 
X_Train, X_Test, Y_Train, Y_Test = train_test_split(
    X, Y, test_size=0.20, random_state=0)

print (Y_Test.shape)
print (X_Test.shape)
"""
#GRID SEARCH
# Establecer los parámetros mediante validación cruzada
param_grid = [{'penalty':['l1'],'C':[1,4,5,10,100,1000],'solver':['liblinear']},
              {'penalty':['l2'],'C':[1,4,5,10,100,1000],'solver':['sag', 'saga','lbfgs']}]



scores = ['precision', 'recall']

for score in scores:
    print(" Ajuste de hiperparámetros para %s" % score)
    print()

    grid = GridSearchCV(
        LogisticRegression(),
        param_grid,
        refit= True,
        n_jobs     = -1,
        cv         = 5, 
        verbose    = 2,
        scoring='%s_micro' % score)
    grid.fit(X_Train, Y_Train)
    print (grid.best_params_)


    print("El mejor conjunto de parámetros que se encuentra en el conjunto de desarrollo:")
    print()
    print(grid.best_params_)
    print()
    print(" Puntuaciones Grid en el conjunto de desarrollo:")
    print()
    means = grid.cv_results_['mean_test_score']
    stds = grid.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, grid.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std , params))
    print()
    
    print("Informe de clasificación detallado:")
    print()
    print("El modelo se entrena en el conjunto de desarrollo completo.")
    print("Las puntuaciones se calculan sobre el conjunto de evaluación completo")
    print()
    y_true, y_pred = Y_Test, grid.predict(X_Test)
    print(classification_report(y_true, y_pred))
    print()




"""



model = LogisticRegression(C= 4, penalty= 'l2',solver='sag')
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)

clf = model.fit(X_Train, Y_Train)
Y_Pred = clf.predict(X_Test)
cm=confusion_matrix(Y_Test, Y_Pred)
print (cm)
# Get metrics
accuracy = cross_val_score(model, X_Train, Y_Train, cv=kfold, scoring='accuracy')
print (accuracy)
print ("Promedio: ", accuracy.mean())
print ("STD: ", accuracy.std())
precision= cross_val_score(model, X_Train, Y_Train, cv=kfold, scoring='precision_weighted')
print (precision)
print ("Promedio: ",precision.mean())
print ("STD: ", precision.std())

recall = cross_val_score(model, X_Train, Y_Train, cv=kfold, scoring='recall')
print (recall)
print ("Promedio: ",recall.mean())
print ("STD: ", recall.std())

f1_macro = cross_val_score(model, X_Train, Y_Train, cv=kfold, scoring='f1_weighted')
print (f1_macro)
print ("Promedio: ",f1_macro.mean())
print ("STD: ", f1_macro.std())



"""

print ("************************************************************************************")
clf = LogisticRegression(C= 4, penalty= 'l2',solver='sag')
#Y_Pred = clf.predict(X_Test)

scores = cross_val_score(clf, X, Y, cv=10, scoring='accuracy')
print (scores)
print("%0.2f de Accuracy con una desviación estándar de %0.2f" % (scores.mean(), scores.std()))

scores = cross_val_score(clf, X, Y, cv=10, scoring='precision_weighted')
print (scores)
print("%0.2f de Precision con una desviación estándar de %0.2f" % (scores.mean(), scores.std()))


scores = cross_val_score(clf, X, Y, cv=10, scoring='recall')
print (scores)
print("%0.2f de Recall con una desviación estándar de %0.2f" % (scores.mean(), scores.std()))

scores = cross_val_score(clf, X, Y, cv=10, scoring='f1_weighted')
print (scores)
print("%0.2f de F1-Score con una desviación estándar de %0.2f" % (scores.mean(), scores.std()))
"""





