import pandas as pd 
from bs4 import BeautifulSoup
import requests
from sqlalchemy import create_engine
import pymysql
from matplotlib import pyplot as plt

engine = create_engine('mysql+pymysql://root:root@127.0.0.1:3306/bruh')

hotelsReviewUrls = []
with open('urlsTripVisor.txt') as webSet:
    for url in webSet:
        hotelsReviewUrls.append(url.replace('\n', ''))


scrapedReviws = []
# script for scraping tripvisor
for url in hotelsReviewUrls:
    getPage = requests.get(url)
    statusCode = getPage.status_code
    
    if(statusCode == 200):
        soup = BeautifulSoup(getPage.text, 'html.parser')
        hotelName = soup.find('h1', class_='_1mTlpMC3').text
        total_reviews = soup.find('span', class_='_33O9dg0j').text
        if price is None:
            price = soup.find('div', class_='CEf5oHnZ')
        else:
            price = soup.find('div', class_='_36QMXqQj')
        total_score = soup.find('span', class_='_3cjYfwwQ').text
        total_score = float(total_score) * 2
        
    for item in soup.findAll('div', class_='_2wrUUKlw _3hFEdNs8'):

        titleReview = item.find('div', class_='glasR4aX').text
        bubble = item.find(class_="ui_bubble_rating").get("class")
        bubble = bubble[1]
        bubble = bubble.replace('bubble_', '')
        bubble = bubble.replace('0', '')
        bubble = int(bubble) * 2
        review = item.find('div', class_='cPQsENeY').text
        review = review.replace('<', '')

        scrapedReviws.append([hotelName, total_reviews, total_score
        ,price.text ,titleReview, review, bubble])
        print("Total Accommodations Scraped: ",len(scrapedReviws))
    else:
        print("Page doesn't respond")

df = pd.DataFrame(scrapedReviws, columns=['hotel_name','totalReviews'
,'totalScore', 'price', 'titleReview', 'review', 'reviewerScore'])

print(df.head())
print(df.info())

df.to_sql('tripvisor', engine, index=True, if_exists='replace')
