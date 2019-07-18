# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 08:46:20 2018

@author: steven.wang
"""

from selenium import webdriver      #动态操作库
from bs4 import BeautifulSoup       #网页解析库
import pandas as pd
import time
import re
## from multiprocessing import Pool

mydata=pd.read_csv('song_info3.csv',encoding='gbk')
site=mydata['url']
song_info={}

driver = webdriver.PhantomJS(executable_path=r'D:\\Users\\steven.wang\\Anaconda3\\phantomjs.exe')
time.sleep(5)
    #无浏览器自动化过程，节省内存使用，简化操作，提高速度
i=0    
for url in site:
    driver.get(url)
    time.sleep(1.5)
    page = driver.page_source
    soup = BeautifulSoup(page, 'html.parser')
        ####title
    try:
        span1 = soup.find_all('div', class_='data__name')       #find其实是字符串
        title = span1[0].find('h1')['title']
    except:
        title=""
        ####singer
    try:
        span2 = soup.find_all('div', class_='data__singer')       
        singer = span2[0].find('a')['title']
    except:
        singer=""
        ####album
    try:
        span3 = soup.find_all('ul', class_='data__info')       
        album = span3[0].find('a')['title']
    except:
        album=""
        ####language
    try:
        span4 = soup.find_all('li', class_='data_info__item js_lan')
        language =str(span4[0]).split('：')[1].strip().split('<')[0]
    except:
        language=""
    try:
        span4 = soup.find_all('li', class_='data_info__item data_info__item--even js_lan')
        language =str(span4[0]).split('：')[1].strip().split('<')[0]
    except:
        pass
        ####genre
    try:
        span5 = soup.find_all('li', class_='data_info__item js_genre data_info__item--even')
        genre =str(span5[0]).split('：')[1].strip().split('<')[0]
    except:
        genre=""
    try:
        span5 = soup.find_all('li', class_='data_info__item js_genre')
        genre =str(span5[0]).split('：')[1].strip().split('<')[0]
    except:
        pass
        ####company
    try:
        span6 = soup.find_all('li', class_='data_info__item data_info__item--even js_company') 
        company=span6[0].get_text().split('：')[-1]
    except:
        company=''
    try:
        span6 = soup.find_all('li', class_='data_info__item js_company') 
        company=span6[0].get_text().split('：')[-1]
    except:
        pass
        ####publish
    try:
        span7 = soup.find_all('li', class_='data_info__item js_public_time data_info__item--even')
        publish =span7[0].get_text().split('：')[-1]
    except:
        publish=''
    try:
        span7 = soup.find_all('li', class_='data_info__item js_public_time')
        publish =span7[0].get_text().split('：')[-1]
    except:
        pass
        ####lyric
    text=soup.find_all('div','lyric__cont_box')
    if  text:
        lyrics = text[0].get_text()
        pattern = re.compile('[\t\n ]+')
        lyrics=re.sub(pattern, ' ', lyrics)
    else:
        lyrics=""
        ####插入信息
    song_info[url]={}
    song_info[url]['title']=title
    song_info[url]['singer']=singer
    song_info[url]['album']=album
    song_info[url]['language']=language
    song_info[url]['genre']=genre
    song_info[url]['company']=company
    song_info[url]['publish']=publish
    song_info[url]['lyrics']=lyrics
    i=i+1
    if i%10 == 0 :
        print (i)
        
    ####写入信息
url=[]
title=[]
singer=[]
album=[]
language=[]
genre=[]
company=[]
publish=[]
lyrics=[]
        
for item in song_info:
    url.append(item)
    title.append(song_info[item]['title'])
    singer.append(song_info[item]['singer'])
    album.append(song_info[item]['album'])
    language.append(song_info[item]['language'])
    genre.append(song_info[item]['genre'])
    company.append(song_info[item]['company'])
    publish.append(song_info[item]['publish'])
    lyrics.append(song_info[item]['lyrics'])
            
data=pd.DataFrame({'url':url,'title':title,'singer':singer,'album':album,'language':language,'genre':genre,'company':company,'publish':publish,'lyrics':lyrics})
data.to_csv("site.csv",index=False)
