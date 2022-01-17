import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import plot_confusion_matrix,accuracy_score,precision_score,recall_score,confusion_matrix,classification_report,f1_score
from sklearn.decomposition import TruncatedSVD
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.naive_bayes import MultinomialNB
from sqlalchemy import create_engine
import pymysql


usr = 'ut8sfdpht7xxh'
password = 'adWAEdxc@aw10)-AAXc24'

engine = create_engine('')

df_chunk = pd.read_sql('call model()',chunksize=10000)

stop = set(stopwords.words('english'))

# lemmatizing the data takes very long on such a big dataset
# watched a few youtube video's to kill the time
# it runs faster on a linux OS(operating system)
def cleanData(dataFrame, indx):
    i=indx
    for sent in dataFrame['Review']:
        lem_list = []
        str1 = ''
        for w in sent.split():
            if((w.isalpha()) & (len(w) > 2)):
                if(w.lower() not in stop):
                    clnd = WordNetLemmatizer().lemmatize(w.lower())
                    lem_list.append(clnd)
        str1 = ' '.join(lem_list)
        dataFrame['cleanedRevs'][i] = str1
        i+=1
        print(i)
    return lem_list

list_of_chunks = []
for chunk in df_chunk:
    chunk['cleanedRevs'] = 'empty'
    chunk.drop(columns='Unnamed: 0', inplace=True)
    cleanData(chunk,chunk.index.start)
    list_of_chunks.append(chunk)


df_concat = pd.concat(list_of_chunks)





