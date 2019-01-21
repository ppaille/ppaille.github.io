# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 22:59:23 2019

@author: ppail_000
"""

from __future__ import print_function

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import dates as dates
from IPython.display import display
import requests
import numpy as np
import datetime
from dateutil.relativedelta import relativedelta
from collections import OrderedDict
import seaborn as sns
import requests




my_key = "my_key"

data_url = "http://api.eia.gov/series/?api_key=" + my_key + "&series_id=NG.N3035MA2.M&;=JSON"
data2_url = "http://api.eia.gov/series/?api_key=" + my_key + "&series_id=NG.N3010MA2.M&;=JSON"

data1 = requests.get(data_url)
data2 = requests.get(data2_url)
 
data = data1.json()
rd = data2.json()
 
 
ind = data['series'][0]['data']
res = rd['series'][0]['data']


    
endi = len(ind)
date = []

for i in range(endi):
    date.append(ind[i][0])
#convert that bad boy to a data frame
df1 = pd.DataFrame(data=date)
df1.columns = ['date']
#loop through the rest of the dict with and put the data in
lenj = len(data)-1

for j in range (lenj):
    ts_series = data['series'][0]['data']
    data = []
    endk = len(ind) 
    for k in range (endk):
        data.append(ind[k][1])
    df1[j] = data
#rename columns and transform date index
df1['date'] = pd.to_datetime(df1['date'], format='%Y%m')
df1.columns = ['date','Industrial Consumption']

#merge dataframes
df3 = pd.concat([df1, df2], keys=['date', 'Industrial Consumption', 'Residential Consumption'])

#set index equal to date column and reverse time sequence
df2 = df2.set_index('date')
date_list = pd.date_range(start='1989-01-01', freq='MS', periods=358)

#df1 set index
df1 = df1.set_index('date')
df1.to_csv('ngi.csv')

#graph data
fig, axarr = plt.subplots(1,1, figsize =(12,8))
df2[['Residential Consumption']].plot(ax=axarr, colormap='Spectral')
axarr.set_ylabel('Natural Gas Consumption')
axarr.set_xlabel('Time');

#convert data into csv for spark
df2.to_csv('ng.csv')


from pyspark.sql import SQLContext
from pyspark.sql.functions import *
from pyspark import SparkContext, SparkConf
conf = SparkConf().setAppName('pyspark')

sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)



sdf = sqlContext.read.csv("C:\\Users\\ppail_000\\Desktop\\testingzone\\ng2.csv",  header=True, inferSchema=True)
net = sqlContext.read.csv("C:\\Users\\ppail_000\\Desktop\\testingzone\\netgen.csv",  header=True, inferSchema=True)

print ('Number of rows: ' , sdf.count())
sdf.printSchema()
sdf.show()

#rename index column from_c0 to date:
sdf = sdf.select(col("_c0").alias("date"), col("Residential Consumption").alias("Res"), col("Industrial Consumption").alias("Ind"))
net = net.select(col("_c0").alias("date"), col("Net Generation").alias("gen"))

sdf1 = sdf[('_c0',1),('Residential Consumption',2),('Industrial Consumption',3)]
sdf2 = pyspark.createDataFrame(sdf1,['date','Residential Consumption', 'Industrial Consumption'])

#convert date-time to date only
k = (net.withColumn("date", date_format('Date', "MM/dd/yyyy"), gen.cast(IntegerType()))
#select months
sdf.select('_c0', date_format('_c0', 'MM').alias('Month')).show() 



joint = sdf.join(net, "date", how='left').select(sdf.date, sdf.Res, sdf.Ind, net.gen).orderBy('date')



group = sdf.groupBy('date')
sdf_resampled = group.select(group.window.start.alias("Start"), group.window.end.alias("End"), "Residential Consumption", "Sum Production").orderBy('Start', ascending=True)
sdf_resampled.printSchema()
sdf_resampled.show()



fig, axarr = plt.subplots(1,1, figsize =(12,8))
joint['Res', 'Ind', 'gen'].plot(ax=axarr)
axarr.set_ylabel('Natural Gas Consumed per Month [MCF/M]')
axarr.set_xlabel('Time');

#visualize with plotly
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode,  plot


joint.toPandas()['date'] = joint.toPandas()['date'].apply(lambda x: datetime.datetime.strptime(x, '%m/%d/%Y'))
joint.toPandas()['date'].str.replace("/","").astype(int)
 
ydate=pd.value_counts(joint.toPandas()['date'], sort=True)
   
trace1 = go.Bar(
    x=joint['date'],
    y=joint.toPandas()['Res'],
    name='Residential Consumption'

 
)
trace2 = go.Bar(
    x=joint['date'],
    y=joint.toPandas()['Ind'],
    name='Industrial Consumption'

)

trace3 = go.Bar(
    x=joint['date'],
    y=joint.toPandas()['gen'],
    name='MWh Generation'

)
data = [trace1, trace2, trace3]
layout = go.Layout(
    barmode='group'
)


fig = go.Figure(data=data, layout=layout)
py.plot(fig)

py.plot(data5)

from plotly.plotly import plot
plot([{"x": joint.toPandas()['date'], "y":[ joint.toPandas()['Res'], joint.toPandas()['Ind']]}])