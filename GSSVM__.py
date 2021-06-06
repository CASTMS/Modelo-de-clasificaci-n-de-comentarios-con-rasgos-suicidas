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


# Importar data

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
vectorizer = TfidfVectorizer(stop_words='english', max_features = 3000)
X = vectorizer.fit_transform(X).toarray()
#tfidf_test = vectorizer.fit_transform(X_Test).toarray()
#tfidf_test = vectorizer.transform(X_Test).toarray()
#print(X_Train)
#print (X)


# Dividir el conjunto de datos en dos partes iguales
X_Train, X_Test, Y_Train, Y_Test = train_test_split(
    X, Y, test_size=0.20, random_state=0)


# Establecer los parámetros mediante validación cruzada
param_grid = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]},
                    {'C': [1,10,100], 'degree' : [2,3,4,5], 'kernel':['poly']}]


scores = ['precision', 'recall']

for score in scores:
    print(" Ajuste de hiperparámetros para %s" % score)
    print()

    grid = GridSearchCV(
        svm.SVC(),
        param_grid,
        #refit= True,
        n_jobs     = -1,
        cv         = 5, 
        verbose    = 2,
        scoring='%s_micro' % score
        )
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
              % (mean, std * 2, params))
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
print ("CROSS VALIDATION SVM")
clf = svm.SVC(kernel='rbf', C=1000, gamma=0.001, random_state=42).fit(X_Train, Y_Train)
Y_Pred = clf.predict(X_Test)
cm=confusion_matrix(Y_Test, Y_Pred)
print (cm)

accuracy = round(((cm[1, 1] + cm[0, 0] )/ (cm[1, 1] + cm[0, 1]+ cm[0, 0]+ cm[1, 0])) * 100, 2)
print("Accuracy: ", accuracy)
# ==============================================================================
accu= accuracy_score(
            y_true    = Y_Test,
            y_pred    = Y_Pred,
            normalize = True
           )
print("")
print(f"El accuracy de test es: {100*accu}%")


#PRECISION = TP/(TP + FP)
precision = round((cm[1, 1] / (cm[1, 1] + cm[0, 1])) * 100, 2)
print("Precision: ", precision)
p=precision_score(Y_Test, Y_Pred)
print (p)


# RECALL = TP/(TP + FN)
recall = round((cm[1, 1] / (cm[1, 1] + cm[1, 0])) * 100, 2)
print("Recall: ", recall)
r=recall_score(Y_Test, Y_Pred)
print (r)
#F1
f1=2*((precision*recall)/(precision+recall))
print ("F1 Score: ", f1)
f=f1_score(Y_Test, Y_Pred)
print (f)



print ("************************************************************************************")
clf = svm.SVC(kernel='rbf', C=1000, gamma=0.001, random_state=42)
scores = cross_val_score(clf, X, Y, cv=5, scoring='accuracy')
print (scores)
print("%0.2f de Accuracy con una desviación estándar de %0.2f" % (scores.mean(), scores.std()))

scores = cross_val_score(clf, X, Y, cv=5, scoring='precision_weighted')
print (scores)
print("%0.2f de Precision con una desviación estándar de %0.2f" % (scores.mean(), scores.std()))

scores = cross_val_score(clf, X, Y, cv=5, scoring='recall')
print (scores)
print("%0.2f de Recall con una desviación estándar de %0.2f" % (scores.mean(), scores.std()))

scores = cross_val_score(clf, X, Y, cv=5, scoring='f1_weighted')
print (scores)
print("%0.2f de F1-Score con una desviación estándar de %0.2f" % (scores.mean(), scores.std()))





from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict 

kf= KFold(n_splits=5)
for train_index, test_index in kf.split(X):
    X_Train, X_Test = X[train_index], X[test_index]
    Y_Train, Y_Test = Y[train_index], Y[test_index]
    clf.fit(X_Train, Y_Train)
    print (confusion_matrix(Y_Test, clf.predict(X_Test)))


"""







