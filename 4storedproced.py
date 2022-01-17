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
from sklearn.metrics import plot_confusion_matrix,accuracy_score,precision_score,recall_score,confusion_matrix,classification_report,f1_score
from sklearn.decomposition import TruncatedSVD
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sqlalchemy import create_engine
from sklearn import preprocessing
import pymysql

engine = create_engine('')

df = pd.read_sql('call model()',engine)

test = df[:154555].reset_index(drop=True)
train = df[154555:df.index.max()].reset_index(drop=True)

train['split'] = 'train'
test['split'] = 'test'

newDf = pd.concat([train, test])

newDf.to_sql('splitted', if_exists='replace', con=engine)