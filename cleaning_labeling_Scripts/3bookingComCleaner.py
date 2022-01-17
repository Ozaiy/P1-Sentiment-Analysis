import pandas as pd 
from bs4 import BeautifulSoup
import requests
from sqlalchemy import create_engine
import pymysql
from matplotlib import pyplot as plt
import numpy as np

engine = create_engine('mysql+pymysql://root:root@127.0.0.1:3306/bruh')

allData = pd.read_sql('call fetchkaggle()', engine)

allData.info()

var_columns_Negative = ['Negative_Review']
var_columns_Positive = ['Positive_Review']

badReviews = pd.DataFrame(allData[var_columns_Negative])
goodReviews = allData[var_columns_Positive]

# Removing Null Values out from all the Bad Reviews
badReviews['totalwords'] = badReviews['Negative_Review'].str.split().str.len()

badReviews.loc[badReviews['totalwords'] <= 2, 'Negative_Review'] = np.nan

badReviews.dropna(
    axis=0,
    how='any',
    thresh=None,
    subset=None,
    inplace=True
)

badReviews.reset_index(inplace=True, drop=True)

badReviews.head()

# Removing Null values out from all Good Reviews
goodReviews['totalwords'] = goodReviews['Positive_Review'].str.split().str.len()

goodReviews.loc[goodReviews['totalwords'] <= 2, 'Positive_Review'] = np.nan

goodReviews.dropna(
    axis=0,
    how='any',
    thresh=None,
    subset=None,
    inplace=True
)

goodReviews.reset_index(inplace=True, drop=True)


goodReviews.head()

badReviews['Label'] = 'Negative'
goodReviews['Label'] = 'Positive'

badReviews.columns = ['Review', 'Total_Words', 'Label']
goodReviews.columns = ['Review', 'Total_Words', 'Label']

goodReviews.head()

badReviews.to_csv('negativeReviewsCleaned.csv')
goodReviews.to_csv('positiveReviewsCleaned.csv')
