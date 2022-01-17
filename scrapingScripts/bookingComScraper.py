from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import pymysql
from selenium.common.exceptions import NoSuchElementException


DRIVER_PATH = 'Desktop\chromedriver.exe'
driver = webdriver.Chrome(executable_path=DRIVER_PATH)
engine = create_engine('')

scrapedReviws = []

url = 'https://www.booking.com/searchresults.html?aid=304142;label=gen173nr-1FCAEoggI46AdIM1gEaKkBiAEBmAExuAEXyAEM2AEB6AEB-AECiAIBqAIDuALllv37BcACAdICJDU0NjI3NGM2LTNkN2ItNDMwOS04MWEyLTk3MjFhZjQxNWUyYtgCBeACAQ;sid=ee5a6741327eec578e914ef601b7f3fd;checkin=2020-10-08;checkout=2020-10-09;dest_id=-2601889;dest_type=city;srpvid=a2067ff3611300a0;ss=London&'

driver.get(url)
driver.maximize_window()
# Getting ready for scraping 

def scrapingPart():
    butn = driver.find_element_by_id('show_reviews_tab')
    driver.implicitly_wait(2)

    average_Score = driver.find_element_by_class_name('bui-review-score__badge')
    average_Scoretxt = average_Score.text

    hotelName = driver.find_element_by_class_name('hp__hotel-name')
    hotelNametxt = hotelName.text

    nextButton = driver.find_elements_by_class_name('bui-pagination__link')

    butn.click()
    driver.implicitly_wait(10)
    revBlock = driver.find_element_by_class_name('review_list')
    driver.implicitly_wait(10)

    revBlock.get_attribute('innerHTML')

    soup = BeautifulSoup(revBlock.get_attribute('innerHTML'), 'html.parser')

    nextButton = driver.find_elements_by_class_name('bui-pagination__link')


    for item in soup.findAll('div', class_='c-review-block'):
        driver.implicitly_wait(5)
        reviewScore = item.find('div', class_='bui-review-score__badge').text
        post = item.find('div', class_='c-review__row')
        negat = item.find('div', class_='c-review__row lalala')

        if negat is None:
            negat = 'NaN'
        else:
            negat = negat.find('span',class_='c-review__body').text

        if post is None:
            post = 'NaN'
        else:
            post = post.find('span',class_='c-review__body').text

        print(post)
        scrapedReviws.append([hotelNametxt, average_Scoretxt, post, negat, reviewScore])

        print("Total Accommodations Scraped:",len(scrapedReviws))
    else:
        print("Page doesn't respond")


links = []
pages = []
for page in driver.find_elements_by_xpath("//a[@class = 'bui-pagination__link sr_pagination_link']"):
    pages.append(page.get_attribute('href'))
    
for i in pages:
    driver.get(i)
    for elmt in driver.find_elements_by_xpath("//a[@class = 'js-sr-hotel-link hotel_name_link url']"):
        links.append(elmt.get_attribute('href'))

print(links)

for i in links:
    driver.get(i)
    scrapingPart()
else:
    print('done')
        
    
        
        

# df = pd.DataFrame(scrapedReviws, columns=['Hotel_Name','Average_Score', 'Positive_Review', 'Negative_Review', 'Reviewer_Score'])


# df.info()
# df.to_sql('bookingcom', engine, index=True, if_exists='replace')

