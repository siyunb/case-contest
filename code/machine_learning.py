# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 10:26:01 2018
Naive Bayes+random forest+svm

@author: situ
"""
import os
import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier  
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn import metrics


os.chdir("E:/graduate/competition/data/")
songs = pd.read_csv("clean_songdata.csv",encoding = "gbk")
songs.head(5)
songs["genre"].value_counts().plot("bar")
genres = list(songs["genre"].value_counts()[:5].index)
otherindex = [i for i in range(songs.shape[0]) if songs["genre"][i] not in genres]
len(otherindex)
songs["genre"][otherindex] = "others"
songs["genre"].value_counts().plot("pie")


def input_data(data,target_name,text_name):
    """
    把读入的文本转化为已分词的语料向量和标签向量
    最好把那些词频低的词、什么无意义的英文去了
    """
    words = data[text_name].tolist()
    tags = data[target_name].tolist()
    return words,tags

def vectorize(words,tags):
    """
    文本向量化
    """
    transformer=TfidfVectorizer()
    data_tfidf=transformer.fit_transform(words)
    transformer2 = TfidfVectorizer(vocabulary = transformer.vocabulary_);                             
    train_words,test_words, train_tags, test_tags = train_test_split(words, tags, test_size=0.3, random_state=42)               
    train_data_tfidf=transformer2.fit_transform(train_words)
    test_data_tfidf=transformer2.fit_transform(test_words)    
    return train_data_tfidf,test_data_tfidf, train_tags, test_tags

def train_clf(train_data_tfidf, train_tags,alpha):
    """
    Naive bayes
    """
    clf = MultinomialNB(alpha=alpha)
    clf.fit(train_data_tfidf, np.asarray(train_tags)) #怎么顺便输出训练错误率？？？
    return clf

def train_rf(train_data_tfidf, train_tags):
    """
    random forest
    """
    rf0 = RandomForestClassifier(n_estimators= 1000, oob_score=True, random_state=10)  
    rf0.fit(train_data_tfidf, train_tags)  
    return rf0

def train_svm(train_data_tfidf, train_tags,iterations=10):
    """
    svm
    """
    svm0 = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=iterations, random_state=42)
    svm0.fit(train_data_tfidf, train_tags)  
    return svm0

def evaluate(actual, pred):
    """
    测试集分类效果
    """
    m_accuracy = metrics.accuracy_score(actual, pred)
    print 'accuracy:{0:.3f}'.format(m_accuracy)
    print "confusion matrix:\n",metrics.confusion_matrix(actual, pred)
    
def main():
    print "input the data"
    words, tags = input_data(songs,"genre","clean_lyrics")
    print "begin to vectorize"
    train_data_tfidf, test_data_tfidf,train_tags,test_tags = vectorize(words,tags)
    print "begin to train NB model"
    clf = train_clf(train_data_tfidf, train_tags,0.001)
    pred1 = clf.predict(train_data_tfidf)
    print "NB train error is ",np.mean(pred1!=train_tags) 
    evaluate(test_tags, clf.predict(test_data_tfidf))
    print "begin to train rf model"
    rf0 = train_rf(train_data_tfidf, train_tags)
    print "rf train obb score is ",rf0.oob_score_  
    evaluate(test_tags,rf0.predict(test_data_tfidf)) 
    print "begin to train svm model"
    svm0 = train_svm(train_data_tfidf, train_tags,10)
    pred2 = svm0.predict(train_data_tfidf)  
    print "svm train error is ",np.mean(pred2!=train_tags)
    evaluate(test_tags,svm0.predict(test_data_tfidf)) 

if __name__ == '__main__':
    main()
    
#模型集成
words, tags = input_data(songs,"genre","clean_lyrics")
train_data_tfidf, test_data_tfidf,train_tags,test_tags = vectorize(words,tags)
p = []
np.random.seed(1994)
for i in np.random.randint(1,100,5):
    print i
    m3 = train_clf(train_data_tfidf, train_tags,0.001)
    p.extend(m3.predict(test_data_tfidf)) 
    
    m2 = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=10, random_state=i)
    m2.fit(train_data_tfidf, train_tags) 
    p.extend(m2.predict(test_data_tfidf)) 
    
    m1 = RandomForestClassifier(n_estimators= 1000, oob_score=True, random_state=i)  
    m1.fit(train_data_tfidf, train_tags) 
    p.extend(m1.predict(test_data_tfidf)) 


p1 = np.array(p)
p1 = p1.reshape(15,758)
p1 =p1.T
from scipy.stats import mode
def getmode(x):
    return mode(x)[0][0]
pre_mode = map(getmode,p1)
len(pre_mode)
np.mean(np.array(pre_mode)== np.array(test_tags))
evaluate(test_tags,pre_mode) 
