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
from sqlalchemy import create_engine
import pymysql
from sklearn.naive_bayes import MultinomialNB

engine = create_engine('')

df_chunk = pd.read_sql('call model()',engine, chunksize=10000)

for chunk in df_chunk:
  chunk.shape


X = df['cleanedRevs']
y = df['Label']

  # Bag of Words (BoW)
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0,test_size=0.3)

vect = CountVectorizer(max_features=1000,binary=True)

X_trainvect = vect.fit_transform(X_train)

y_testvect = vect.transform(X_test)

counts = df["Label"].value_counts()
print(counts)

print("\nPredicting only -1 = {:.2f}% accuracy".format(counts[0] / sum(counts) * 100))

multiBayes = MultinomialNB()

multiBayes.fit(X_trainvect, y_train)

multiBayes.score(X_trainvect, y_train)


y_pred = multiBayes.predict(y_testvect)

print("Confusion Matrix knn:\n", confusion_matrix(y_test, y_pred))
print("Accuracy for naiveBayes: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))

print('---------------')
print('KNN algho')

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_trainvect, y_train)

pred = knn.predict(y_testvect)

print("Confusion Matrix naive:\n",confusion_matrix(y_test, pred))
# plot_confusion_matrix(knn, X_tr, y_tr)

print("Accuracy for KNN: {:.2f}%".format(accuracy_score(y_test, pred) * 100))

# print('---------------')
# print('naive bayes algho')

# naiveBayes = GaussianNB()

# naiveBayes.fit(X_tr.toarray(), y_tr)
# pred = naiveBayes.predict(X_te.toarray())

# print(confusion_matrix(pred, y_te))
# plot_confusion_matrix(naiveBayes, X_train.toarray(), y_train)

# print('accuracy for Gaussian naiveBayes:', accuracy_score(y_te, pred))


# Knearest(vec_train_text,vec_test_text,y_train,y_test)

# naiveBays(vec_train_text,vec_test_text,y_train,y_test)



# X_tr_vectorized = vect.transform(X_1)
# x_cv_vectorized = vect.transform(X_test)

# runKNN(X_tr_vectorized,x_cv_vectorized,y_tr,y_cv,'Bag of Words')

# # Applying TFIDF
# vect_tfidf = TfidfVectorizer(min_df = 5).fit(df['cleanedRevs'])
# X_tr_vectorized = vect_tfidf.transform(X_tr)
# x_cv_vectorized = vect_tfidf.transform(X_cv)

# runKNN(X_tr_vectorized,x_cv_vectorized,y_tr,y_cv,'TF-IDF')

# df.to_csv('checkfor.csv', if_exists='replace')

# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis("off")
# plt.show()

