# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 12:59:58 2018

@author: steven.wang
"""
from numpy import *
from gensim import *
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from keras.optimizers import *
from keras.constraints import maxnorm




data_clean=pd.read_csv("clean_songdata.csv",encoding='gbk')
data11=data_clean[data_clean['genre']==data_clean['genre']]
data11['genre']=data11['genre'].str.replace('amp;','')
types=['Pop','Rap/Hip Hop','Country','R&B','Dance']
data11['genre']=[i if i in types else 'Other' for i in data11['genre']]
Y = data11['genre']
texts=[]
for item in data11['clean_lyrics']:
    item=item.split()
    text=[word for word in item]
    texts.append(text)

dictionary = corpora.Dictionary(texts)
## 删去频率低于5次的词
dictionary.filter_extremes(no_below=5)
corpus = [dictionary.doc2bow(text) for text in texts]
tfidf_model = models.TfidfModel(corpus)
corpus_tfidf = tfidf_model[corpus]
X = matutils.corpus2dense(corpus, num_terms=len(dictionary)).T

# encode class values as integers
encoder = LabelEncoder()
encoded_Y = encoder.fit_transform(Y)
# convert integers to dummy variables (one hot encoding)
dummy_y = np_utils.to_categorical(encoded_Y)

input_dim=X.shape[1]
output_dim=dummy_y.shape[1]
X_train, X_test, Y_train, Y_test = train_test_split(X, dummy_y, test_size=0.3)
# define model structure
def baseline_model(optimizer='adam'):
    model = Sequential()
    model.add(Dense(output_dim=10, input_dim=input_dim))
    model.add(Dense(output_dim=output_dim, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

estimator = KerasClassifier(build_fn=baseline_model, verbose=0)
# splitting data into training set and test set. If random_state is set to an integer, the split datasets are fixed.
Y_index = where(Y_test==1)[1]
estimator.fit(X_train, Y_train)

# make predictions
pred = estimator.predict(X_test)
accuracy=sum(Y_index ==pred)/len(pred)

# inverse numeric variables to initial categorical labels
init_lables = encoder.inverse_transform(pred)

# k-fold cross-validate
seed = 42
np.random.seed(seed)
results=[]
for i in range(5):
    X_train, X_test, Y_train, Y_test = train_test_split(X, dummy_y, test_size=0.3)
    estimator.fit(X_train, Y_train)
    kfold = KFold(n_splits=3, shuffle=True)
    result = cross_val_score(estimator, X, dummy_y, cv=kfold).mean()
    results.append(result)
mean(results)
kfold = KFold(n_splits=5, shuffle=True)
results = cross_val_score(estimator, X, dummy_y, cv=kfold)
results.mean() 
## 准确率0.54

## 优化算法
optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adam', 'Nadam']
#optimizer = ['Adagrad', 'Nadam']
acc=[]
for opt in optimizer:
    estimator = KerasClassifier(build_fn=baseline_model,optimizer=opt,verbose=0)
    results=[]
    for i in range(4):
        X_train, X_test, Y_train, Y_test = train_test_split(X, dummy_y, test_size=0.3)
        estimator.fit(X_train, Y_train)
        kfold = KFold(n_splits=3, shuffle=True)
        result = cross_val_score(estimator, X, dummy_y, cv=kfold).mean()
        results.append(result)
    accuracy=mean(results)
    acc.append(accuracy)
    print("accuracy: %f, opt: %s" % (accuracy, opt))
print (max(acc))
print (argmax(acc))
## 最优参数：Adagrad(0.557),Nadam(0.548)

import matplotlib.pyplot as plt
from pylab import *  
mpl.rcParams['font.sans-serif'] = ['SimHei']
#创建柱状图，设置颜色，透明度和外边框颜色
plt.bar([1,2,3,4,5],acc,color='#99CC01',alpha=0.8)
#设置x轴标签
plt.rc('font', family='STXihei', size=15)
plt.xlabel(u'优化算法')
#设置y周标签
plt.ylabel(u'预测准确率')
#设置图表标题
plt.title(u'不同优化算法的准确率')
#设置数据分类名称
a=np.array([1,2,3,4,5])
plt.xticks(a,optimizer)


## 隐藏层神经元数量
def baseline_model(neurons=10,opt='Adagrad'):
    model = Sequential()
    model.add(Dense(output_dim=neurons, input_dim=input_dim))
    #model.add(Dropout(0.1))
    model.add(Dense(output_dim=output_dim, activation='softmax'))
    #optimizer = Nadam(lr=0.01)
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

neurons = [5*i for i in range(1,6)]
acc=[]
for ne in neurons:
    results=[]
    for i in range(4):
        X_train, X_test, Y_train, Y_test = train_test_split(X, dummy_y, test_size=0.3)
        estimator = KerasClassifier(build_fn=baseline_model, neurons=ne, verbose=0)
        estimator.fit(X_train, Y_train)
        kfold = KFold(n_splits=3, shuffle=True)
        result = cross_val_score(estimator, X, dummy_y, cv=kfold)
        results.append(result)
    accuracy=mean(results)
    acc.append(accuracy)
    print("accuracy: %f, ne: %d" % (accuracy, ne))
print (max(acc))
print (argmax(acc))
## 最优参数20(0.567)

import matplotlib.pyplot as plt
from pylab import *  
mpl.rcParams['font.sans-serif'] = ['SimHei']

plt.plot(neurons,acc,"g-",color='#99CC01',linewidth=3,marker='o',markeredgewidth=3)   #在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度）  
plt.xlabel(u"神经元数目") #X轴标签  
plt.ylabel(u"预测准确率")  #Y轴标签  
plt.title(u"不同神经元数量的准确率") #图标题  


## 隐藏层数目
def baseline_model(num=1):
    model = Sequential()
    model.add(Dense(output_dim=10, input_dim=input_dim))
    #model.add(Dropout(0.1))
    while num>1:
        model.add(Dense(output_dim=10))
        #model.add(Dropout(0.1))
        num=num-1
    model.add(Dense(output_dim=output_dim, activation='softmax'))
    #optimizer = Nadam(lr=0.01)
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='Adagrad', metrics=['accuracy'])
    return model

nums = range(1,6)
acc=[]
for n in nums:
    results=[]
    for i in range(4):
        X_train, X_test, Y_train, Y_test = train_test_split(X, dummy_y, test_size=0.3)
        estimator = KerasClassifier(build_fn=baseline_model, num=n,verbose=0)
        estimator.fit(X_train, Y_train)
        kfold = KFold(n_splits=3, shuffle=True)
        result = cross_val_score(estimator, X, dummy_y, cv=kfold)
        results.append(result)
    accuracy=mean(results)
    acc.append(accuracy)
    print("accuracy: %f, n: %d" % (accuracy, n))
print (max(acc))
print (argmax(acc))

import matplotlib.pyplot as plt
from pylab import *  
mpl.rcParams['font.sans-serif'] = ['SimHei']

plt.plot(nums,acc,"g-",color='#99CC01',linewidth=3,marker='o',markeredgewidth=3)   #在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度）  
plt.xlabel(u"隐藏层数目") #X轴标签  
plt.ylabel(u"预测准确率")  #Y轴标签  
plt.title(u"不同隐藏层的准确率") #图标题  


## 初始化方法
def baseline_model(init_mode='uniform'):
    model = Sequential()
    model.add(Dense(output_dim=10, init=init_mode, input_dim=input_dim))
    model.add(Dense(output_dim=output_dim, init=init_mode, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='Adagrad', metrics=['accuracy'])
    return model
init_mode = ['uniform', 'lecun_uniform', 'normal',  'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
#param_grid = dict(init_mode=init_mode)
acc=[]
for im in init_mode:
    results=[]
    for i in range(4):
        X_train, X_test, Y_train, Y_test = train_test_split(X, dummy_y, test_size=0.3)
        estimator = KerasClassifier(build_fn=baseline_model, init_mode=im,verbose=0)
        estimator.fit(X_train, Y_train)
        kfold = KFold(n_splits=3, shuffle=True)
        result = cross_val_score(estimator, X, dummy_y, cv=kfold)
        results.append(result)
    accuracy=mean(results)
    acc.append(accuracy)
    print("accuracy: %f, im: %s" % (accuracy, im))
print (max(acc))
print (argmax(acc))
## 最优参数：uniform(0.574) 

## 激活函数
def baseline_model(activation='relu'):
    model = Sequential()
    model.add(Dense(output_dim=15, init='lecun_uniform', input_dim=input_dim, activation=activation))
    model.add(Dense(output_dim=output_dim, init='lecun_uniform', activation='softmax'))
    #optimizer = Nadam(lr=0.01)
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='Adagrad', metrics=['accuracy'])
    return model
activation = ['softplus',  'relu', 'tanh', 'linear']
# param_grid = dict(activation=activation)
acc=[]
for ac in activation:
    results=[]
    for i in range(4):
        X_train, X_test, Y_train, Y_test = train_test_split(X, dummy_y, test_size=0.3)
        estimator = KerasClassifier(build_fn=baseline_model, activation =ac,verbose=0)
        estimator.fit(X_train, Y_train)
        kfold = KFold(n_splits=3, shuffle=True)
        result = cross_val_score(estimator, X, dummy_y, cv=kfold)
        results.append(result)
    accuracy=mean(results)
    acc.append(accuracy)
    print("accuracy: %f, ac: %s" % (accuracy, ac))
print (max(acc))
print (argmax(acc))
## 最优参数：tanh(0.584) 没有改进
import matplotlib.pyplot as plt
from pylab import *  
mpl.rcParams['font.sans-serif'] = ['SimHei']
#创建柱状图，设置颜色，透明度和外边框颜色
plt.bar([1,2,3,4],acc,color='#99CC01',alpha=0.8)
#设置x轴标签
plt.rc('font', family='STXihei', size=15)
plt.xlabel(u'激活函数')
#设置y周标签
plt.ylabel(u'预测准确率')
plt.axis([0.5,4.5,0.5,0.6]) 
#设置图表标题
plt.title(u'不同激活函数的准确率')
#设置数据分类名称
a=np.array([1,2,3,4])
plt.xticks(a,activation)

## 神经元数量
def baseline_model(neurons=10):
    model = Sequential()
    model.add(Dense(output_dim=neurons, init='lecun_uniform',input_dim=input_dim,activation='tanh'))
    #model.add(Dropout(0.1))
    model.add(Dense(output_dim=output_dim, init='lecun_uniform',activation='softmax'))
    #optimizer = Nadam(lr=0.01)
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='Nadam', metrics=['accuracy'])
    return model
neurons = [15,20,25,30]
acc=[]
for ne in neurons:
    results=[]
    for i in range(3):
        X_train, X_test, Y_train, Y_test = train_test_split(X, dummy_y, test_size=0.3)
        estimator = KerasClassifier(build_fn=baseline_model, neurons=ne, verbose=0)
        estimator.fit(X_train, Y_train)
        kfold = KFold(n_splits=3, shuffle=True)
        result = cross_val_score(estimator, X, dummy_y, cv=kfold)
        results.append(result)
    accuracy=mean(results)
    acc.append(accuracy)
    print("accuracy: %f, ne: %d" % (accuracy, ne))
print (max(acc))
print (argmax(acc))
## 最优参数20(0.567)


## 正则化参数和范数约束
def baseline_model(dropout_rate=0.0, weight_constraint=0):
    model = Sequential()
    model.add(Dense(output_dim=15, init='lecun_uniform', input_dim=input_dim,activation='tanh'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(output_dim=output_dim, init='lecun_uniform',activation='softmax'))
    #optimizer = Nadam(lr=0.01)
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='Adagrad', metrics=['accuracy'])
    return model

#weight_constraint = [6]
dropout_rate = [ 0,0.05,0.1,0.15]
acc=[]
for dr in dropout_rate:
    results=[]
    for i in range(4):
        X_train, X_test, Y_train, Y_test = train_test_split(X, dummy_y, test_size=0.3)
        estimator = KerasClassifier(build_fn=baseline_model, dropout_rate=dr, verbose=0)
        estimator.fit(X_train, Y_train)
        kfold = KFold(n_splits=3, shuffle=True)
        result = cross_val_score(estimator, X, dummy_y, cv=kfold)
        results.append(result)
    accuracy=mean(results)
    acc.append(accuracy)
    print("accuracy: %f,  dr: %f" % (accuracy,dr))
print (max(acc))
print (argmax(acc))
## 最优参数：dr:0.05,wc:6（0.58）没有改进

## 范数约束
def baseline_model( dr=0,weight_constraint=0):
    model = Sequential()
    model.add(Dense(output_dim=15, init='lecun_uniform', input_dim=input_dim,
                    W_constraint=maxnorm(weight_constraint),activation='tanh'))
    model.add(Dropout(dr))
    model.add(Dense(output_dim=output_dim, init='lecun_uniform',activation='softmax'))
    #optimizer = Nadam(lr=0.01)
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='Adagrad', metrics=['accuracy'])
    return model

weight_constraint = [4,5,6]
dropout_rate = [ 0.05,0.1,0.15]
acc=[]
for wc in weight_constraint[1:2]:
    results=[]
    for i in range(4):
        X_train, X_test, Y_train, Y_test = train_test_split(X, dummy_y, test_size=0.3)
        estimator = KerasClassifier(build_fn=baseline_model, weight_constraint=wc, verbose=0)
        estimator.fit(X_train, Y_train)
        kfold = KFold(n_splits=3, shuffle=True)
        result = cross_val_score(estimator, X, dummy_y, cv=kfold)
        results.append(result)
    accuracy=mean(results)
    acc.append(accuracy)
    print("accuracy: %f,  dr: %f" % (accuracy,wc))
print (max(acc))
print (argmax(acc))

import matplotlib.pyplot as plt
from pylab import *  
mpl.rcParams['font.sans-serif'] = ['SimHei']

plt.plot(weight_constraint,acc,"g-",color='#99CC01',linewidth=3,marker='o',markeredgewidth=3)   #在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度）  
plt.axis([4.5,7.5,0.55,0.6]) 
plt.xlabel(u"正则化参数") #X轴标签  
plt.ylabel(u"预测准确率")  #Y轴标签  
plt.title(u"不同正则化参数的准确率") #图标题  

## 学习速率和动量因子
def baseline_model(learn_rate=0.01):
    model = Sequential()
    model.add(Dense(output_dim=15,init='lecun_uniform', input_dim=input_dim,
                    W_constraint=maxnorm(6), activation='tanh'))
    model.add(Dropout(0.05))
    model.add(Dense(output_dim=output_dim, init='lecun_uniform',activation='softmax'))
    optimizer = Adagrad(lr=learn_rate)
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

learn_rate = [0.005,0.01,0.015,0.02]
decays=[0.2,0.5,0.8]
# momentum = [0.2, 0.4, 0.6, 0.8]
# param_grid = dict(learn_rate=learn_rate, momentum=momentum)
acc=[]
for lr in learn_rate:
    estimator = KerasClassifier(build_fn=baseline_model, learn_rate=lr,verbose=0)
    results=[]
    for i in range(4):
        X_train, X_test, Y_train, Y_test = train_test_split(X, dummy_y, test_size=0.3)
        estimator.fit(X_train, Y_train)
        kfold = KFold(n_splits=3, shuffle=True)
        result = cross_val_score(estimator, X, dummy_y, cv=kfold)
        results.append(result)
    accuracy=mean(results)
    acc.append(accuracy)
    print("accuracy: %f, lr: %s" % (accuracy, lr))
print (max(acc))
print (argmax(acc))
## 最优学习速率：0.01(0.584）没有改进


## 批尺寸和训练epochs
def baseline_model():
    model = Sequential()
    model.add(Dense(output_dim=15,init='lecun_uniform', input_dim=input_dim,
                    W_constraint=maxnorm(6), activation='tanh'))
    model.add(Dense(output_dim=output_dim, init='lecun_uniform',activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='Adagrad', metrics=['accuracy'])
    return model
acc=[]
batch_size = [5,10,20,30]
epochs = [10,20,30,40]
for batch in batch_size:
    results=[]
    for i in range(4):
        X_train, X_test, Y_train, Y_test = train_test_split(X, dummy_y, test_size=0.3)
        estimator = KerasClassifier(build_fn=baseline_model, 
                                    batch_size=batch, verbose=0)
        estimator.fit(X_train, Y_train)
        kfold = KFold(n_splits=3, shuffle=True)
        result = cross_val_score(estimator, X, dummy_y, cv=kfold)
        results.append(result)
    accuracy=mean(results)
    acc.append(accuracy)
    print("accuracy: %f, batch: %d" % (accuracy, batch))
print (max(acc))
print (argmax(acc))
## 最优参数为：batch=10,epoch=30 (0.58)

## epoch
def baseline_model():
    model = Sequential()
    model.add(Dense(output_dim=15,init='lecun_uniform', input_dim=input_dim,
                    W_constraint=maxnorm(6), activation='tanh'))
    model.add(Dense(output_dim=output_dim, init='lecun_uniform',activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='Adagrad', metrics=['accuracy'])
    return model
acc=[]
batch_size = [5,10,20]
epochs = [10,20,30,40]
for batch in batch_size:
    for epoch in epochs:
        results=[]
        for i in range(4):
            X_train, X_test, Y_train, Y_test = train_test_split(X, dummy_y, test_size=0.3)
            estimator = KerasClassifier(build_fn=baseline_model, 
                                    batch_size=batch, nb_epoch=epoch,verbose=0)
            estimator.fit(X_train, Y_train)
            kfold = KFold(n_splits=3, shuffle=True)
            result = cross_val_score(estimator, X, dummy_y, cv=kfold)
            results.append(result)
        accuracy=mean(results)
        acc.append(accuracy)
        print("accuracy: %f, batch: %d, epoch: %d" % (accuracy,batch,epoch))
print (max(acc))
print (argmax(acc))
## 最优参数为：batch=10,epoch=30 (0.58)


## 隐藏层层数
def baseline_model(num=0):
    model = Sequential()
    model.add(Dense(output_dim=15, init='lecun_uniform', input_dim=input_dim,
                     activation='tanh'))
    #model.add(Dropout(0.1))
    while num>1:
        model.add(Dense(output_dim=10))
        #model.add(Dropout(0.1))
        num=num-1
    model.add(Dense(output_dim=output_dim, init='lecun_uniform', activation='softmax'))
    #optimizer = Nadam(lr=0.01)
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='Adagrad', metrics=['accuracy'])
    return model

nums = [1,2,4,6,8]
acc=[]
for n in nums:
    results=[]
    for i in range(4):
        X_train, X_test, Y_train, Y_test = train_test_split(X, dummy_y, test_size=0.3)
        estimator = KerasClassifier(build_fn=baseline_model, num=n,verbose=0)
        estimator.fit(X_train, Y_train)
        kfold = KFold(n_splits=3, shuffle=True)
        result = cross_val_score(estimator, X, dummy_y, cv=kfold)
        results.append(result)
    accuracy=mean(results)
    acc.append(accuracy)
    print("accuracy: %f, n: %d" % (accuracy, n))
print (max(acc))
print (argmax(acc))
## 1层 58.1% 没有改进


## 综合考虑
X_train, X_test, Y_train, Y_test = train_test_split(X, dummy_y, test_size=0.3)
# define model structure
def baseline_model(optimizer='Adagrad'):
    model = Sequential()
    model.add(Dense(output_dim=20, init='lecun_uniform',input_dim=input_dim,
                    activation='tanh'))
    model.add(Dense(output_dim=output_dim, init='lecun_uniform',activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

estimator = KerasClassifier(build_fn=baseline_model, batch_size=5, 
                                nb_epoch=30,verbose=0)
# splitting data into training set and test set. If random_state is set to an integer, the split datasets are fixed.
Y_index = where(Y_test==1)[1]
estimator.fit(X_train, Y_train)

# make predictions
pred = estimator.predict(X_test)
sum(Y_index ==pred)/len(pred)
accuracy=sum(Y_index ==pred)/len(pred)

def show_table(y_true,y_pred):
    import numpy as np
    import pandas as pd
    from sklearn.metrics import confusion_matrix
    matrix=confusion_matrix(y_true,y_pred)
    level=np.unique(y_true).tolist()
    Index=['True_'+str(content) for content in level]
    columns=['pred_'+str(content) for content in level]
    return(pd.DataFrame(matrix,index=Index,columns=columns))

show_table(Y_index,pred)

from sklearn.metrics import confusion_matrix
confusion_matrix(Y_index,pred)
# k-fold cross-validate
results=[]
for i in range(5):
    X_train, X_test, Y_train, Y_test = train_test_split(X, dummy_y, test_size=0.3)
    estimator = KerasClassifier(build_fn=baseline_model,batch_size=10, 
                                nb_epoch=20, verbose=0)
    estimator.fit(X_train, Y_train)
    kfold = KFold(n_splits=3, shuffle=True)
    result = cross_val_score(estimator, X, dummy_y, cv=kfold).mean()
    results.append(result)
mean(results)





