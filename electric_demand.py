# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from dateutil.relativedelta import relativedelta
import seaborn as sns
import statsmodels.api as sm  
from statsmodels.tsa.stattools import acf  
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.seasonal import seasonal_decompose

#dateparse = pd.datetime.strptime(dates, '%Y/%m/%d')
df = pd.read_csv('C:\\Users\\ppail_000\\Desktop\\rthl_data.csv', parse_dates=['Date'], index_col='Date')

#Resample hourly data to daily data
ddf = df.resample('24H').mean()
ddm = ddf.resample('30D').mean()

df.index.name='Date'
df.reset_index(inplace=True)
#df.drop(df.index[114], inplace=True)




start = datetime.datetime.strptime("2012-12-01", "%Y-%m-%d")
date_list = [start + relativedelta(days=x) for x in range(1,541)]
df['Date'] =date_list
df.set_index(['Date'], inplace=False)
df.index.name='Date'

df.columns= ['Mwh']
df['MWh'] = df.MWh.apply(lambda x: int(x)*100)

df.MWh.plot(figsize=(12,8), title= 'Real Time MWh Demand', fontsize=14)

#Seasonality Visualization
#use ddf.values if timeseries is smaller than 1yr
decomposition = seasonal_decompose(ddf, freq=12)  
fig = plt.figure()  
fig = decomposition.plot()  
fig.set_size_inches(15, 8)

#create tuple for ADF:
data1 = ddf.iloc[:,0].values
data2 = ddm.iloc[:,0].values

from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean = pd.rolling_mean(timeseries, window=7)
    rolstd = pd.rolling_std(timeseries, window=7)

    #Plot rolling statistics:
    fig = plt.figure(figsize=(12, 8))
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()
    
    #Perform Dickey-Fuller test:
   # print 'Results of Dickey-Fuller Test:'
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
        print(dfoutput)
    
#Create First Difference to eliminate trend:
ddm['first_difference'] = ddm.MWh - ddm.MWh.shift(1)
test_stationarity(ddm.first_difference.dropna(inplace=False))

#Remove Seasonality:
ddm['seasonal_difference'] = ddm.MWh - ddm.MWh.shift(12)
test_stationarity(ddm.seasonal_difference.dropna(inplace=False))

#First difference with seasonality removed:
ddm['seasonality_first_difference'] = ddm.first_difference - ddm.first_difference.shift(12)
test_stationarity(ddm.seasonality_first_difference.dropna(inplace=False))

#Plot ACF and PACF charts
fig = sm.graphics.tsa.plot_acf(ddm.seasonality_first_difference.iloc[13:], lags=40, ax=ax1)

#Create SARIMA Model:
mod = sm.tsa.statespace.SARIMAX(ddm.MWh, trend='n', order=(1,1,1), seasonal_order=(0,1,1,12))
results = mod.fit()
print(results.summary())


#Forecasting:
ddm['forecast'] = results.predict(start = 74, end= 94, dynamic= True)  
ddm[['MWh', 'forecast']].plot(figsize=(12, 8))

#Visualize actual results:
npredict =ddm.MWh['2018'].shape[0]
fig, ax = plt.subplots(figsize=(12,6))
npre = 12
ax.set(title='Reat Time MWh System Demand', xlabel='Date', ylabel='MWh')
ax.plot(ddm.index[-npredict-npre+1:], ddm.ix[-npredict-npre+1:, 'MWh'], 'o', label='Observed')
ax.plot(ddm.index[-npredict-npre+1:], ddm.ix[-npredict-npre+1:, 'forecast'], 'g', label='Dynamic forecast')
legend = ax.legend(loc='lower right')
legend.get_frame().set_facecolor('w')

#Future Forecast
start = datetime.datetime.strptime("2018-12-01", "%Y-%m-%d")
date_list = [start + relativedelta(months=x) for x in range(0,12)]
future = pd.DataFrame(index=date_list, columns= ddm.columns)
ddm = pd.concat([ddm, future])

ddm['forecast'] = results.predict(start = 74, end = 94, dynamic= True)
ddm[['MWh', 'forecast']].ix[-24:].plot(figsize=(12, 8)) 


