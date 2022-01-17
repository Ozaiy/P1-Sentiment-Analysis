import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import string
import nltk
import seaborn as sns

from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.metrics import roc_curve,r2_score,plot_confusion_matrix,accuracy_score,precision_score,recall_score,confusion_matrix,classification_report,f1_score
from sklearn.decomposition import TruncatedSVD
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sqlalchemy import create_engine
from sklearn import preprocessing
import pymysql
from sklearn.feature_extraction.text import TfidfTransformer


engine = create_engine('')

# train = pd.read_sql("call selectsplit('train')",engine)
# test = pd.read_sql("call selectsplit('test')",engine)

# X_train = train['cleanedRevs']
# X_test = test['cleanedRevs']

# y_train = train['Label']
# y_test = test['Label']

naivebayes = MultinomialNB()
svc = LinearSVC()
forest = RandomForestClassifier()

df = pd.read_sql("call model()",engine)



# i=0
# lispredictions=[]
# for df in df_chunk:
#     X = df['cleanedRevs']
#     y = df['Label']
#     i+=1
#     print(i)
#     # Bag of Words (BoW)
#     X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0,test_size=0.2)

#     vect = CountVectorizer(max_features=12000)

#     X_train = vect.fit_transform(X_train)

#     X_test = vect.transform(X_test)
        
#     naivebayes.fit(X_train, y_train)
#     prednaive = naivebayes.predict(X_test)
#     print("MultinomialNaiveBayes")
#     print("avg Accuracy:", accuracy_score(y_test, prednaive) * 100)
#     print(confusion_matrix(y_test, prednaive))
#     print('\n')

X = df['cleanedRevs']
y = df['Label']

  # Bag of Words (BoW)
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0,test_size=0.2)

vect = CountVectorizer(max_features=12000)

X_train = vect.fit_transform(X_train)

X_test = vect.transform(X_test)

naivebayes.fit(X_train, y_train)
prednaive = naivebayes.predict(X_test)
print("MultinomialNaiveBayes")
print("avg Accuracy:", accuracy_score(y_test, prednaive) * 100)
print(confusion_matrix(y_test, prednaive))
print('\n')

forest.fit(X_train, y_train)
predforest = forest.predict(X_test)
print("RandomForest")
print("avg Accuracy:", accuracy_score(y_test, predforest) * 100)
print(confusion_matrix(y_test, predforest))
print('\n')

svc.fit(X_train, y_train)
predsvc = svc.predict(X_test)
print("LinearSVC")
print("avg Accuracy:", accuracy_score(y_test, predsvc) * 100)
print(confusion_matrix(y_test, predsvc))
print('\n')






# X_train, X_test, y_train, y_test

# 

