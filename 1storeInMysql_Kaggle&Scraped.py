import pandas as pd 
from bs4 import BeautifulSoup
import requests
from sqlalchemy import create_engine
import pymysql
from matplotlib import pyplot as plt


engine = create_engine('')

eigenData = pd.read_csv('base_csvs/eigenReviews.csv')
tripvisor = pd.read_sql('select * from tripvisor', engine)
hotelCSV = pd.read_csv('base_csvs/Hotel_Reviews.csv')
bookingCom = pd.read_sql('select * from bookingcom', engine)

eigenData.shape
tripvisor.shape
bookingCom.shape

tripvisor['source'] = 'tripvisor'
hotelCSV['source'] = 'kaggleCSV'
eigenData['source'] = 'eigenData'
bookingCom['source'] = 'bookingcom'

# df = pd.concat([tripvisor, bookingCom, hotelCSV, eigenData], ignore_index=True)

# df.to_sql('combinedset', engine, index=True, if_exists='replace')

tripvisor = pd.read_sql("call choosesource('tripvisor')", engine)
kaggle = pd.read_sql("call choosesource('kaggleCSV')", engine)

kaggle = kaggle[:10000]

kaggle['Reviewer_Score']
kaggle['Reviewer_Nationality']





plt.show()