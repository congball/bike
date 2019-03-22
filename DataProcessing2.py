#!/usr/bin/python
#-*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib
import seaborn as sns
from pylab import mpl
import matplotlib.pyplot as plt
from matplotlib.font_manager import _rebuild
_rebuild()
import calendar
from datetime import datetime

mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

train = pd.read_csv('/Users/liucong/Desktop/node/Data-anlysis/Bike/data/train.csv')
print(train.info())
#检查异常值
print(train.describe())

test = pd.read_csv('/Users/liucong/Desktop/node/Data-anlysis/Bike/data/test.csv')
print(test.info())
print(test.describe())

# #count数值差异大，观察其密度分布
# fig = plt.figure()
# ax = fig.add_subplot(1,1,1)
# fig.set_size_inches(6,5)
# sns.distplot(train['count'])
# ax.set(xlabel='count', title='Distribution of count')
# plt.show()

#排除掉3个标准差以外的数据
train_WithOutliers = train[np.abs(train['count'] - train['count'].mean()) <= (3*train['count'].std())]          #abs()返回数据的绝对值
print(train_WithOutliers.shape)
print(train_WithOutliers.describe())
#去掉3个标准差以后的数据分布
# fig = plt.figure()
# ax1 = fig.add_subplot(1,2,1)
# ax2 = fig.add_subplot(1,2,2)
# fig.set_size_inches(12,5)
# sns.distplot(train_WithOutliers['count'], ax=ax1)
# sns.distplot(train['count'], ax=ax2)
# ax1.set(xlabel = 'count', title='Distribution of count without outliers')
# ax2.set(xlabel = 'registered' , title = 'Distribution of count')
# plt.show()

# yLabels = train_WithOutliers['count']
# yLabels_log = np.log(yLabels)
# sns.distplot(yLabels_log)
# plt.show()

Bike_data = pd.concat([train_WithOutliers,test], ignore_index=True)
print(Bike_data.shape)
print(Bike_data.head())

#把datetime拆分为日期、时段、年份、月份、星期五列
Bike_data['date'] = Bike_data.datetime.apply(lambda c: c.split()[0])
Bike_data['hour'] = Bike_data.datetime.apply(lambda c: c.split()[1].split(':')[0]).astype('int')
Bike_data['year'] = Bike_data.datetime.apply(lambda c: c.split()[0].split('-')[0]).astype('int')
Bike_data['month'] = Bike_data.datetime.apply(lambda c: c.split()[0].split('-')[1]).astype('int')
Bike_data['weekday'] = Bike_data.date.apply(lambda c: datetime.strptime(c,'%Y-%m-%d').isoweekday())
print(Bike_data.head())


# #查看temp、atemp、humidity、windspeed四项的分布  【数值型数据，记得查看其分布】
# fig, axes = plt.subplots(2,2)
# fig.set_size_inches(12,10)
# sns.distplot(Bike_data['temp'], ax=axes[0,0])
# sns.distplot(Bike_data['atemp'], ax=axes[0,1])
# sns.distplot(Bike_data['humidity'], ax=axes[1,0])
# sns.distplot(Bike_data['windspeed'], ax=axes[1,1])
# axes[0,0].set(xlabel = 'temp', title = 'Distribution of temp')
# axes[0,1].set(xlabel = 'atemp', title = 'Distribution of atemp')
# axes[1,0].set(xlabel = 'humidity', title = 'Distribution of humidity')
# axes[1,1].set(xlabel = 'windspeed', title = 'Distribution of windspeed')
# plt.show()

print(Bike_data[Bike_data['windspeed'] !=0]['windspeed'].describe())

#采用随机森林填充风速
from sklearn.ensemble import RandomForestRegressor
Bike_data["windspeed_rfr"] = Bike_data['windspeed']
#将数据分成风速等于0和不等于0两部分
dataWind0 = Bike_data[Bike_data["windspeed_rfr"] == 0]
dataWindNot0 = Bike_data[Bike_data["windspeed_rfr"] != 0]
#选定模型
rfModel_wind = RandomForestRegressor(n_estimators=1000, random_state=42)
#选定特征值
windColumns = ['season', 'weather', 'humidity', 'month', 'temp', 'year', 'atemp']
#将风速不等于0的数据作为训练数据集,fit到RandomForestRegressor中
rfModel_wind.fit(dataWindNot0[windColumns], dataWindNot0['windspeed_rfr'])
#通过训练好的模型预测风速
wind0Values = rfModel_wind.predict(X = dataWind0[windColumns])
#填充到风速为0的数据中
dataWind0.loc[:, 'windspeed_rfr'] = wind0Values
#连接两部分数据
Bike_data = dataWindNot0.append(dataWind0)
Bike_data.reset_index(inplace=True)
Bike_data.drop('index', inplace=True, axis=1)

# fig, axes = plt.subplots(2,2)
# fig.set_size_inches(12,10)              #set_size_inches用于设置图形的尺寸
# sns.distplot(Bike_data['temp'],ax=axes[0,0])
# sns.distplot(Bike_data['atemp'],ax=axes[0,1])
# sns.distplot(Bike_data['humidity'],ax=axes[1,0])
# sns.distplot(Bike_data['windspeed_rfr'],ax=axes[1,1])
#
# axes[0,0].set(xlabel='temp',title="Distribution of temp")
# axes[0,1].set(xlabel='atemp',title="Distribution of atemp")
# axes[1,0].set(xlabel='humidity',title="Distribution of humidity")
# axes[1,1].set(xlabel='windspeed',title="Distribution of windspeed")
#plt.show()


#整体观察
#pairplot()散点矩阵图
# sns.pairplot(Bike_data, x_vars=['holiday', 'workingday', 'weather', 'season', 'weekday', 'hour', 'windspeed_rfr', 'humidity', 'temp', 'atemp'],
#              y_vars=['casual', 'registered', 'count'], plot_kws={'alpha':0.1})
# plt.show()

#相关性矩阵
corrDf = Bike_data.corr()
print(corrDf['count'].sort_values(ascending=False))

# #逐项展示各个特征对目标值count的影响
# workingday_df = Bike_data[Bike_data['workingday'] == 1]
# workingday_df = workingday_df.groupby(['hour'], as_index=True).agg({'casual':'mean','registered':'mean','count':'mean'})        #agg() 数据聚合  #as_index=True 以有索引的形式返回数据
#
# nworkingday_df = Bike_data[Bike_data['workingday'] == 0]
# nworkingday_df = nworkingday_df.groupby(['hour'], as_index=True).agg({'casual':'mean','registered':'mean','count':'mean'})
#
# fig, axes = plt.subplots(1,2,sharey=True)           #sharey=True表示各子图共用一个y轴标签
# workingday_df.plot(figsize=(15,5), title='The average number of rentals initiated per hour in the workingday', ax=axes[0])
# nworkingday_df.plot(figsize=(15,5), title='The average number og rentals initiated per hour in the nworkingday', ax=axes[1])
# plt.show()

#temp对count的影响
#数据按照天汇总，取一天中气温的中位数
temp_df = Bike_data.groupby(['date','weekday'], as_index=False).agg({'year':'mean','month':'mean','temp':'median'})           #as_index=False 以无索引的形式返回数据
#删除存在缺失的数据，防止折线图有断裂
temp_df.dropna(axis=0, how='any', inplace=True)         # 参数 how = ‘any’ 有1个缺失值的行就删除
#按照天统计数据波动较大，于是按月取日平均值进行统计
temp_month = temp_df.groupby(['year','month'], as_index=False).agg({'weekday':'mean','temp':'median'})
#将按照天求和统计的数据的日期转换成datetime格式
temp_df['date'] = pd.to_datetime(temp_df['date'])

temp_month.rename(columns = {'weekday':'day'},inplace=True)
temp_month['date'] = pd.to_datetime(temp_month[['year','month','day']])

fig = plt.figure(figsize=(18,6))
ax = fig.add_subplot(1,1,1)                     #第1行、第1列、第1个位置
# #折线图展示count随时间的走势
# plt.plot(temp_df['date'], temp_df['temp'], lineWidth = 1.3, label='Daily average')
# ax.set_title('Change trend of average temperature per day in two years')
# plt.plot(temp_month['date'], temp_month['temp'], marker='o', lineWidth=1.3, label='Monthly average')
# ax.legend()         #显示图例
# plt.show()
#
# #按温度取count平均值
# temp_rantals = Bike_data.groupby(['temp'], as_index=True).agg({'casual':'mean','registered':'mean','count':'mean'})
# temp_rantals.plot(title='The average number of rentals initiated per hour changes with the temperature')
# plt.show()


#humidity对count的影响
#观察humidity的走势
humidity_df = Bike_data.groupby('date', as_index=False).agg({'humidity':'mean'})
humidity_df['date'] = pd.to_datetime(humidity_df['date'])

humidity_df = humidity_df.set_index('date')
humidity_month = Bike_data.groupby(['year','month'], as_index=False).agg({'weekday':'mean','humidity':'mean'})
humidity_month.rename(columns={'weekday':'day'}, inplace=True)
humidity_month['date'] = pd.to_datetime(humidity_month[['year','month','day']])

# fig = plt.figure(figsize=(18,6))
# ax = fig.add_subplot(1,1,1)
# plt.plot(humidity_df.index, humidity_df['humidity'], lineWidth=1.3, label='Daily average')
# plt.plot(humidity_month['date'], humidity_month['humidity'], marker='o', lineWidth=1.3, label='Monthly average')
# ax.legend()
# ax.set_title('Change trend of average humidity per day in two days')
# plt.show()
#
# humidity_rentals = Bike_data.groupby(['humidity'], as_index=True).agg({'casual':'mean','registered':'mean','count':'mean'})
# humidity_rentals.plot(title='Average number of rentals initiated per hour in different humidity')
# plt.show()


# #年份、月份对count的影响
# #按天汇总
# count_df = Bike_data.groupby(['date','weekday'], as_index=False).agg({'year':'mean','month':'mean','casual':'sum','registered':'sum','count':'sum'})
# count_df.dropna(axis=0, how='any', inplace=True)
# #按月取平均值
# count_month = count_df.groupby(['year','month'], as_index=False).agg({'weekday':'min','casual':'mean','registered':'mean','count':'mean'})
# count_df['date'] = pd.to_datetime(count_df['date'])
# count_month.rename(columns={'weekday':'day'},inplace=True)
# count_month['date'] = pd.to_datetime(count_month[['year','month','day']])

# fig = plt.figure(figsize=(16,8))
# ax = fig.add_subplot(1,1,1)
# plt.plot(count_df['date'],count_df['count'],lineWidth=1.3, label='Daily average')
# ax.set_title('Change trend of average number of rentals initiated per day in two years')
# plt.plot(count_month['date'],count_month['count'],marker='o',lineWidth=1.5,label='Monthly average')
# plt.show()


# #认真完成其可视化（此部分未完成# ）
day_df = Bike_data.groupby('date').agg({'year':'mean','season':'mean','casual':'sum','registered':'sum','count':'sum','temp':'mean','atemp':'mean'})
season_df = Bike_data.groupby(['year','season'],as_index=True).agg({'casual':'mean','registered':'mean','count':'mean'})
temp_df = Bike_data.groupby(['year','season'],as_index=True).agg({'temp':'mean','atemp':'mean'})

# fig = plt.figure(figsize=(16,8))
# ax = fig.add_subplot(1,1,1)
# season_df.plot(title='number of rentals of different seasons')
# ax = fig.add_subplot(2,1,1)
# temp_df.plot(title='the temp change in different seasons')
# plt.show()

#查看天气对出行的影响
count_weather = Bike_data.groupby('weather')
result = count_weather[['casual','registered','count']].count()
print(result)
weather_df = Bike_data.groupby('weather', as_index=True).agg({'casual':'mean','registered':'mean'})
# weather_df.plot.bar(stacked=True, title='number  of rentals in different weather')
# plt.show()


#日期对出行的影响
day_df = Bike_data.groupby(['date']).agg({'casual':'sum','registered':'sum','count':'sum','workingday':'mean','weekday':'mean','holiday':'mean','year':'mean'})
print(day_df.head())

number_pei = day_df[['casual','registered']].mean()
print(number_pei)
# plt.axes(aspect='equal')
# plt.pie(number_pei, labels=['casual','registered'], autopct='%1.1f%%', pctdistance=0.6, labeldistance=0.8, radius=1)
# plt.title('Casual or registered in the total lease')
# plt.show()

#工作日/非工作日对出行的影响
workingday_df = day_df.groupby(['workingday'], as_index=True).agg({'casual':'mean','registered':'mean'})
workingday_df_0 = workingday_df.loc[0]
workingday_df_1 = workingday_df.loc[1]

# fig = plt.figure(figsize=(8,6))
# plt.subplots_adjust(hspace=0.5, wspace=0.2)         #设置子图表问题
# grid = plt.GridSpec(2,2, wspace=0.5, hspace=0.5)        #设置子图坐标轴  对齐
#
# plt.subplot2grid((2,2),(1,0),rowspan=2)
# width = 0.3
#
# p1 = plt.bar(workingday_df.index, workingday_df['casual'], width)
# p2 = plt.bar(workingday_df.index, workingday_df['registered'], width, bottom=workingday_df['casual'])
# plt.title("每天的平均用户数量")
# plt.xticks([0,1], ('nonworking day', 'working day'), rotation=20)
# plt.legend((p1[0], p2[0]), ('casual', 'registered'))
# plt.show()


#选择特征值
#将多类别型数据使用one-hot转化成多个二分型类别
dummis_month = pd.get_dummies(Bike_data['month'], prefix='month')
dummis_season = pd.get_dummies(Bike_data['season'], prefix='season')
dummis_weather = pd.get_dummies(Bike_data['weather'], prefix='weather')
dummis_year = pd.get_dummies(Bike_data['year'], prefix='year')
#和原来的数据连接起来
Bike_data = pd.concat([Bike_data,dummis_month,dummis_season,dummis_weather,dummis_year], axis=1)      #axis=1为行

#分离训练集和测试集
dataTrain = Bike_data[pd.notnull(Bike_data['count'])]
dataTest = Bike_data[pd.notnull(Bike_data['count'])].sort_values(by=['datetime'])
datetimecol = dataTest['datetime']
yLabels = dataTrain['count']
yLabels_log = np.log(yLabels)

#丢弃不要的列
dropFeatures = ['casual','count','datetime','date','registered','windspeed','atemp','month','season','weather','year']
dataTrain = dataTrain.drop(dropFeatures, axis=1)
dataTest = dataTest.drop(dropFeatures, axis=1)

#选择模型、训练模型
rfModel = RandomForestRegressor(n_estimators=1000, random_state=42)
rfModel.fit(dataTrain, yLabels_log)
preds = rfModel.predict(X=dataTrain)

#预测测试集数据
predsTest = rfModel.predict(X=dataTest)
submission = pd.DataFrame({'datetime':datetimecol, 'count':[max(0,x) for x in np.exp(predsTest)]})
submission.to_csv('/Users/liucong/Desktop/node/Data-anlysis/Bike/data/bike_predictions.csv', index=False)