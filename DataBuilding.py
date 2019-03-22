#!/usr/bin/python
#-*- coding:utf-8 -*-

import pandas as pd
import numpy as np
from pylab import mpl
import calendar
from datetime import datetime
import matplotlib
from  matplotlib.font_manager import _rebuild
_rebuild()

mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

traindata = pd.read_csv('/Users/liucong/Desktop/node/Data-anlysis/Bike/data/train.csv')
traindata.drop('holiday_label',axis=1)
testdata = pd.read_csv('/Users/liucong/Desktop/node/Data-anlysis/Bike/data/test.csv')
row_train = traindata.shape[0]
now_train = traindata.shape[1]
row_test = testdata.shape[0]
now_test = testdata.shape[1]
fulldata = pd.concat([traindata, testdata], axis=0)
print(fulldata.info())
print(traindata.info())

#字段处理
#train中
fulldata['date'] = fulldata.datetime.map(lambda c: c.split()[0])
fulldata['hour'] = fulldata.datetime.map(lambda c: c.split()[1].split(':')[0]).astype('int')
fulldata['year'] = fulldata.datetime.map(lambda  c: c.split()[0].split('-')[0]).astype('int')
fulldata['month'] = fulldata.datetime.map(lambda  c: c.split()[0].split('-')[1]).astype('int')
fulldata['week'] = fulldata.date.map(lambda c: datetime.strptime(c, '%Y-%m-%d').isoweekday()).astype('int')

traindata = traindata.drop('holiday_label', axis=1)


# #变量映射
# traindata['season_label'] = traindata.season.map({1:"Spring",2:"Summer",3:"Fall",4:"Winter"})
# traindata['weather_label'] = traindata.weather.map({1:"sunny",2:"cloudy",3:"rainly",4:"bad-day"})
# traindata['holiday_label'] = traindata.holiday.map({0:"non-holiday",1:"holiday"})
#
# testdata['season_label'] = testdata.season.map({1:"Spring",2:"Summer",3:"Fall",4:"Winter"})
# testdata['weather_label'] = testdata.weather.map({1:"sunny",2:"cloudy",3:"rainly",4:"bad-day"})
# testdata['holiday_label'] = testdata.holiday.map({0:"non-holiday",1:"holiday"})


# #建模分析
# #去掉不做分析的量
# dropName = [ 'datetime','atemp', 'casual', 'registered']
# data = fulldata.drop(dropName, axis=1)
# data['temp'] = data['temp'].astype('int')
# data['windspeed'] = data['windspeed'].astype('int')
# biketraindf = data.loc[:row_train-1,:]
# biketraindf['count'] = biketraindf['count'].astype('int')
# biketraindf['holiday'] = biketraindf['holiday'].astype('int')
# print(biketraindf.head())
# source_y = biketraindf.loc[0:row_train-1, 'count']
# source_X = biketraindf.drop('count',axis=1)
# pre_X = data.iloc[row_train:, 1:11]
# print(pre_X.head())
#
#
#
# #选择LinearRegression建模分析
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# import warnings
# warnings.filterwarnings('ignore')
#
# source_ylog = np.log1p(source_y)        #取log(y+1)
# print(source_ylog.head())
# train_X, test_X, train_y, test_y = train_test_split(source_X, source_ylog, train_size=.7)
#
# #LinearRegression
# lr = LinearRegression()
# lr.fit(train_X, train_y)
# lr_score = lr.score(test_X, test_y)
# print('lr分数:',lr_score)
# pre_y = np.expm1(lr.predict(pre_X))
# pre_y = pre_y.astype(int)
# datetime = data.loc[:,'datetime']
# preDF = pd.DataFrame({'datetime':datetime, 'count':pre_y})
# print(preDF.shape())

