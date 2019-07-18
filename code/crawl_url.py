# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 08:44:02 2018

@author: steven.wang
"""
##载入相关库
import sys
import requests
#import urllib
import time
from bs4 import BeautifulSoup
import pandas as pd
import re
import json

##设置首页url和头部信息，防止ip被封
headers = {
    'User-Agent':'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.80 Safari/537.36 Core/1.47.933.400 QQBrowser/9.4.8699.400',
}
url = 'http://c.y.qq.com//v8/fcg-bin/fcg_v8_toplist_cp.fcg?tpl=3&page=detail&date=2018_11&topid=108&type=global&song_begin=0&song_num=100 \
    &g_tk=245426252&jsonpCallback=MusicJsonCallbacktoplist&loginUin=404011463&hostUin=0&format=jsonp&inCharset=utf8&outCharset=utf-8 \
    &notice=0&platform=yqq&needNewCode=0'
#date=2018_10  2018年第10周的榜单，可更改
#topid=108 billb榜单对应id是108
#song_num=100  每页显示100条歌曲的信息

url='https://y.qq.com/n/yqq/song/003aAYrm3GE0Ac.html'
data = requests.get(url, headers=headers)
data.encoding
html=data.content
dic_html = html[26:(len(html)-1)] #提取json文本
#json.loads(dic_html)["songlist"][98]["data"]["songmid"]
len(json.loads(dic_html)["songlist"])
json.loads(dic_html)["songlist"][0]["data"]
json.loads(dic_html)["songlist"][0]["data"]['songname']
for i in range(len(json.loads(dic_html)["songlist"])):
    songmid=json.loads(dic_html)["songlist"][i]["data"]
   
year=2018
week=11
song_info={}
for j in range(325):
    ## 这几周的榜单数据缺失
    if ((year==2017) and (week==34)) or ((year==2016) and (week==4)) or ((year==2015) and (week==23)):
        week=week-1
    if year>2014:
        url = 'http://c.y.qq.com//v8/fcg-bin/fcg_v8_toplist_cp.fcg?tpl=3&page=detail&date='+str(year)+'_'+str(week)+'&topid=108&type=global&song_begin=0&song_num=100&g_tk=245426252&jsonpCallback=MusicJsonCallbacktoplist&loginUin=404011463&hostUin=0&format=jsonp&inCharset=utf8&outCharset=utf-8&notice=0&platform=yqq&needNewCode=0'
    else:
        url = 'http://szc.y.qq.com//v8/fcg-bin/fcg_v8_toplist_cp.fcg?tpl=3&page=detail&date='+str(year)+'_'+str(week)+'&topid=108&type=global&song_begin=0&song_num=100&g_tk=245426252&jsonpCallback=MusicJsonCallbacktoplist&loginUin=404011463&hostUin=0&format=jsonp&inCharset=utf8&outCharset=utf-8&notice=0&platform=yqq&needNewCode=0'
    page = requests.get(url, headers=headers)
    html=page.content
    dic_html = html[26:(len(html)-1)] #提取json文本
    for i in range(len(json.loads(dic_html)["songlist"])):
        data=json.loads(dic_html)["songlist"][i]["data"]
        songid=data['songid']
        if songid in song_info.keys():
            song_info[songid]['start']=str(year)+'_'+str(week)
        else:
            song_info[songid]={}
            song_info[songid]['name']=data['songname']
            song_info[songid]['songmid']=data['songmid']
            song_info[songid]['url']="https://y.qq.com/n/yqq/song/"+data['songmid']+".html"
            song_info[songid]['albummid']=data['albummid']
            song_info[songid]['albumname']=data['albumname']
            song_info[songid]['singer']=data['singer'][0]['name']
            song_info[songid]['start']=str(year)+'_'+str(week)
            song_info[songid]['end']=str(year)+'_'+str(week)
    if week==1:
        if year==2012:
            break
        else:
            week=52
            year=year-1
    else:
        week=week-1
    if j%10==0:
        print (j)
        
len(song_info)


id=[]
name=[]
songmid=[]
url=[]
ablummid=[]
ablum=[]
singer=[]
start=[]
end=[]
for item in song_info:
    id.append(item)
    name.append(song_info[item]['name'])
    songmid.append(song_info[item]['songmid'])
    url.append(song_info[item]['url'])
    ablummid.append(song_info[item]['albummid'])
    ablum.append(song_info[item]['albumname'])
    singer.append(song_info[item]['singer'])
    start.append(song_info[item]['start'])
    end.append(song_info[item]['end'])
    
data=pd.DataFrame({'id':id,'songmid':songmid,'name':name,'url':url,'ablummid':ablummid,'ablum':ablum,'singer':singer,'start':start,'end':end})
data.to_csv("song_info3.csv",index=False)
data.to_csv("song_info2.csv",encoding='utf-8')


data=pd.read_csv("song_info.csv")
last=[]
for i in range(data.shape[0]):
    start=data['start'][i].split('_')
    end=data['end'][i].split('_')
    count=(int(end[0])-int(start[0]))*52+int(end[1])-int(start[1])
    
    last.append(count)

index=[last.index(x) for x in sorted(last,reverse=True)[:30]]
index=[]
for i in range(30):
    for j in range(len(last)):
        if last[j]==sorted(last,reverse=True)[i]:
            if j not in index:
                index.append(j)
data.iloc[index,[3,5,1,6]]


last in sorted(last,reverse=True)[:15]






