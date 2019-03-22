#！/usr/bin/python
#-*- coding:utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import missingno as msno
import seaborn as sns
from pylab import mpl
from matplotlib.font_manager import _rebuild
_rebuild()
sns.set()
#日期处理包导入
import calendar
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report



mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

df = pd.read_csv('/Users/liucong/Desktop/node/Data-anlysis/Bike/data/bike.csv')
print(df.info())
print(df.shape)             #数据集的大小

describe = df.describe()
describe.to_csv('/Users/liucong/Desktop/node/Data-anlysis/Bike/data/describe.csv')
print(df.describe())

#1、日期字段的处理
#提取date
df["date"] = df.datetime.apply(lambda x: x.split()[0])
#print(df["date"])

#提取hour
df["hour"] = df.datetime.apply(lambda x: x.split()[1].split(":")[0])
dateString = df.datetime[1].split()[0]

#提取weekday
df["weekday"] = df.date.apply(lambda dateString: calendar.day_name[datetime.strptime(dateString,"%Y-%m-%d").weekday()])
#print(df['weekday'])

#提取month
df["month"] = df.date.apply(lambda dateString: calendar.month_name[datetime.strptime(dateString,"%Y-%m-%d").month])
#print(df['month'])

#2、变量映射处理,将属于定性变量的数值取值，做映射处理，转换为描述性取值
#季节映射处理
df["season_label"] = df.season.map({1:"Spring",2:"Summer",3:"Fall",4:"Winter"})

#天气映射处理
df["weather_label"] = df.weather.map({1:"sunny",2:"cloudy",3:"rainly",4:"bad-day"})

#是否节假日映射处理
df["holiday_map"] = df["holiday"].map({0:"non-holiday",1:"holiday"})
#df.to_csv('/Users/liucong/Desktop/node/Data-anlysis/Bike/data/bike.csv')


#可视化查看缺失值
#msno.matrix(df,labels=True)


#3、数据分析(数据探索和可视化)
#关系热力图
# correlation = df[["temp","atemp","casual","registered","humidity","windspeed","count"]].corr()
# mask = np.array(correlation)
# mask[np.tril_indices_from(mask)] = False                  #tril()返回一个三角矩阵
# fig,ax = plt.subplots()
# fig.set_size_inches(20,15)
# sns.heatmap(correlation, mask=mask, vmax=.9, square=True, annot=True)
# print(correlation)
# plt.show()


# #按不同因素划分的租车人数的分布情况
# #设置绘图格式和画图大小
# fig, axes = plt.subplots(nrows=2, ncols=2)   #axes是matplotlib.axes._subplots.AxesSubplot 这个类型的,我们可以理解为这是一个子plot
# fig.set_size_inches(12,10)
# #第一个子图，租车人数分布的箱线图
# sns.boxplot(data=df, y="count", orient="v", ax= axes[0][0])
# axes[0][0].set(ylabel="Count", title="Box Plot On Count")
#
# #租车人数季节分布箱线图
# sns.boxplot(data=df, y="count", x="season", orient="v", ax=axes[0][1])                #orient="v|h"用于控制图像使其水平、竖直显示
# axes[0][1].set(ylabel="Count", xlabel="Season", title="Box Plot On Count Across Season")
#
# sns.boxplot(data=df, y="count", x="hour", orient="v", ax=axes[1][0])
# axes[1][0].set(ylabel="Count", xlabel="Hour", title="Box Plot On Count Across Hour")
# sns.boxplot(data=df, y="count", x="workingday", orient="v", ax=axes[1][1])
# axes[1][1].set(ylabel="Count", xlabel="Workingday", title="Box Plot On Count Across Workingday")
#
#
# # #湿度&温度对应的租车人数
# #温度、湿度离散化
# df["temp_band"] = pd.cut(df["temp"],5)
# df["humidity_band"] = pd.cut(df["humidity"],5)
#
# df["holiday_map"] = df["holiday"].map({0:"non-holiday",1:"holiday"})
# #FaceGrid用于绘制各变量之间的关系图
# sns.FacetGrid(data=df, row="humidity_band", size=3, aspect=2).\
# map(sns.barplot,'temp_band','count','holiday_map',palette='deep', ci=None).\
# add_legend()                                #用来添加轴坐标，显示图中的标签
# plt.show()
#
#
# #风速对应的租车人数
# df["windspeed_band"] = pd.cut(df["windspeed"],5)
# df["humidity_band"] = pd.cut(df["humidity"],5)
#
# df["holiday_map"] = df["holiday"].map({0:"non-holiday",1:"holiday"})
# sns.FacetGrid(data=df, row="humidity_band", size=3, aspect=2).\
# map(sns.barplot, 'windspeed_band', 'count', 'holiday_map', palette='deep', ci=None).\
# add_legend()
#
#
#
# #不同季节下每小时平均租车人数的变化
# sns.FacetGrid(data=df, size=8, aspect=1.5).\
# map(sns.pointplot, 'hour', 'count', 'season_label', palette="deep", ci=None).\
# add_legend()
#
#
# #按照星期，每小时平均租车人数的变化
# sns.FacetGrid(data=df, size=8, aspect=1.5).\
# map(sns.pointplot, 'hour', 'count', 'weekday', palette="deep", ci=None).\
# add_legend()


#按照天气，每个月的平均租车人数变化
#palette表示调色盘
# sns.FacetGrid(data=df, size=8, aspect=3).\
# map(sns.pointplot, 'month','count','weather_label', palette="deep", ci=None).\
# add_legend()
# #plt.show()
#
# #用决策树增加模型的预测能力
# #提取关键特征
# x = df[['hour']]
# y = df['count']
# #print(x.info())
#
#
# #数据分割
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=33)
# #使用特征转换器进行特征抽取
# vec = DictVectorizer()
# x_train = vec.fit_transform(x_train.to_dict(orient="record"))
# x_test = vec.transform(x_test.to_dict(orient="record"))
#
# #训练模型，进行建模
# #初始化决策树分类器
# dtc = DecisionTreeClassifier()
# #训练
# dtc.fit(x_train, y_train)
# #预测，保存结果
# y_predict = dtc.predict(x_test)
#
#
# #模型评估
# print("准确度",dtc.score(x_test, y_test))
# # print("其他指标",classification_report(y_predict, y_test, target_names=['registered']))









