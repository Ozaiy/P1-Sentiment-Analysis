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
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import sklearn.linear_model as linear_model



from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import SnowballStemmer
from sklearn import metrics
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.metrics import auc,roc_auc_score,roc_curve,r2_score,plot_confusion_matrix,accuracy_score,precision_score,recall_score,confusion_matrix,classification_report,f1_score
from sklearn.decomposition import TruncatedSVD
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sqlalchemy import create_engine
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer

import pickle

import pymysql
from sklearn.feature_extraction.text import TfidfTransformer
import wordcloud as wrd 

engine = create_engine('')

# df_negative = pd.read_sql('call model("Negative")',con=engine)
# df_positive = pd.read_sql('call model("Positive")',con=engine)
# df = pd.concat([df_negative,df_positive])
# df = df.sample(frac=1).reset_index(drop=True)

df = pd.read_sql('call model("alldata")',engine)
df = df[:100000]

# df.to_sql('')
# wordcloud = wrd.WordCloud().generate(text)
# text = " ".join(review for review in df.cleanedRevs)
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis("off")
# plt.show()

df.head()

df = df[:100000]
df.groupby('Label')
df['Label'].value_counts()


# negative en positive reviews gelijk stellen
g = df.groupby('Label')
g = pd.DataFrame(g.apply(lambda x: x.sample(g.size().min()).reset_index(drop=True)))
df = g

df['Label'].value_counts()

naivebayes = MultinomialNB()
knn = LogisticRegression(random_state=0)
forest = KNeighborsClassifier()

# Init a dictionary for storing results of each run for each model

X = df['cleanedRevs']
y = df['Label']

  # Bag of Words (BoW)
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0,test_size=0.2)

vect = CountVectorizer()

X_train = vect.fit_transform(X_train)

X_test = vect.transform(X_test)

def naiveBmulti():
  naivebayes.fit(X_train, y_train)
  proba_naive = naivebayes.predict_proba(X_test)[:,1]
  pred_naive = naivebayes.predict(X_test)
  auc_naive = roc_auc_score(y_test, proba_naive)
  print("MultinomialNaiveBayes")
  print('auc score', auc_naive)
  print("Accuracy:", accuracy_score(y_test, pred_naive) * 100)
  print(confusion_matrix(y_test, pred_naive))
  print(classification_report(y_test,pred_naive))
  print('\n')
  fpr, tpr, _ = roc_curve(y_test, proba_naive, pos_label='Positive')
  plt.plot([0, 1], [0, 1], linestyle='--')
  plt.plot(fpr,tpr,color='darkorange')
  plt.show()
  plot_confusion_matrix(naivebayes, X_test,y_test)
  plt.show()
  
  print('\n')
  

def rndmForest():
  forest.fit(X_train, y_train)
  predforest = forest.predict(X_test)
  proba_forest = forest.predict_proba(X_test)[:,1]
  auc_forest = roc_auc_score(y_test, proba_forest)
  print("RandomForest")
  print('auc score:', auc_forest)
  print("Accuracy:", accuracy_score(y_test, predforest) * 100)
  print(confusion_matrix(y_test, predforest))
  print(classification_report(y_test,predforest))
  print('\n')
  fpr, tpr, _ = roc_curve(y_test, proba_forest, pos_label='Positive')
  plt.plot([0, 1], [0, 1], linestyle='--')
  plt.plot(fpr,tpr,color='darkorange')
  plt.show()
  plot_confusion_matrix(forest, X_test, y_test)
  plt.show()
  print('\n')
  
def knnmodl():
  knn.fit(X_train, y_train)
  predknn = knn.predict(X_test)
  predprob = knn.predict_proba(X_test)[:,1]
  auc_knn = roc_auc_score(y_test, predprob)
  print("K nearest neighbor")
  print('auc score:', auc_knn)
  print("Accuracy:", accuracy_score(y_test, predknn) * 100)
  print(confusion_matrix(y_test, predknn))
  print(classification_report(y_test, predknn))
  print('\n')
  fpr, tpr, _ = roc_curve(y_test, predprob, pos_label='Positive')
  plt.plot([0, 1], [0, 1], linestyle='--')
  plt.plot(fpr,tpr,color='darkorange')
  plt.show()
  plot_confusion_matrix(knn, X_test, y_test)
  plt.show()
  print('\n')
  
knnmodl()
# rndmForest()
naiveBmulti()
