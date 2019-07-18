# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 10:37:17 2018
分词

@author: situ
"""

import numpy as np
import pandas as pd
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer #词形变化

os.chdir("E:/graduate/competion/data")
songdata0 = pd.read_csv("site03.csv",encoding="gbk")
songdata0.head(5)
#去除歌词为空的行
songdata1 = songdata0[songdata0["lyrics"].notnull()]
#去除非英文
songdata2 = songdata1[songdata1["language"]=="英语"]

#去除重复样本
columns=[column for column in songdata2]
columns.remove("url")
songdata3 = songdata2.drop_duplicates(columns)
dup_lyrics = np.unique(songdata3["lyrics"][songdata3["lyrics"].duplicated()])

def del_dup_sample(songdata3,dup_lyrics_i):
    dup_songs = songdata3[songdata3["lyrics"]==dup_lyrics_i]
    index = list(dup_songs.index)
    columns=[column for column in dup_songs]
    new_sample = pd.DataFrame(np.zeros([1,len(columns)]),columns=columns)
    for var in columns:
        if len(dup_songs[var].dropna())>0:
            new_sample[var] = np.array(dup_songs[var].dropna())[0]
        else:
            new_sample[var] = np.nan
    return new_sample,index

for i in range(len(dup_lyrics)):
#for i in range(5):
    dup_lyrics_i = dup_lyrics[i]
    new_sample,index_i = del_dup_sample(songdata3,dup_lyrics_i)
    if i==0:
        new_sample_all = new_sample.copy()
        index_all = index_i.copy()
    else:
        new_sample_all = pd.concat([new_sample_all,new_sample])
        index_all.extend(index_i)
new_sample_all
index_all

lyrics_not_dup = songdata3.drop(index_all)
rm_dup_songdata = pd.concat([new_sample_all,lyrics_not_dup])
#为了行索引问题，保存成csv又导入
#rm_dup_songdata.to_csv("rm_dup_songdata.csv",index=False)

#是否有歌词太短的：没有
os.chdir("E:/graduate/competition/data")
songdata4 = pd.read_csv("rm_dup_songdata.csv",encoding="gbk")
songdata4["lyric_len"] = [len(lyric_i) for lyric_i in songdata4["lyrics"]]
songdata4.sort_values("lyric_len", ascending=True).to_csv("lyric_len_sort.csv",index=False)

#example for 去除歌词中的数字、非英文
#l = songdata4["lyrics"][32] #含中文、数字
#l = songdata4["lyrics"][835] #含日文
#l = l.encode("utf-8")
#l = l.decode("utf-8")
#l=re.sub("\d","",l) #去除数字
#re.sub(u'[\u4e00-\u9fa5]',"",l) #去除中文
#re.sub(u'[\u3000-\u303f\u4e00-\u9fa5]', "", l) #去除中文及中文标点

#去除数字、非英文字符、停用词、转换成小写、词形还原
wnl = WordNetLemmatizer()
def clean_lyric(lyric):
    lyric = lyric.encode("utf-8").decode("utf-8")
    lyric = re.sub("\d","",lyric)
    lyric_seg = [re.sub(u'\W', "", i) for i in nltk.word_tokenize(lyric)] 
    space_len = lyric_seg.count(u"")
    for i in range(space_len):
        lyric_seg.remove(u'')
    filtered = [w.lower() for w in lyric_seg if w not in stopwords.words('english') and 3<=len(w)] 
    lemmatized= [wnl.lemmatize(w) for w in filtered] #词形还原
    return " ".join(lemmatized)

songdata4["clean_lyrics"] = map(clean_lyric,songdata4["lyrics"])
songdata4.head(5)


songdata4.to_csv("clean_songdata.csv",index=False,encoding = "gbk")
