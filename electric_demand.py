# -*- coding: utf-8 -*-
"""
Spyder Editor

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

#Visualize
df.MWh.plot(figsize=(12,8), title= 'Real Time MWh Demand', fontsize=14)

#Seasonality Visualization
decomposition = seasonal_decompose(ddf.values, freq=12)  
fig = plt.figure()  
fig = decomposition.plot()  
fig.set_size_inches(15, 8)

#create tuple for ADF:
data1 = ddf.iloc[:,0].values


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
ddf['first_difference'] = ddf.MWh - ddf.MWh.shift(1)
test_stationarity(ddf.first_difference.dropna(inplace=False))

#Remove Seasonality:
ddf['seasonal_difference'] = ddf.MWh - ddf.MWh.shift(12)
test_stationarity(ddf.seasonal_difference.dropna(inplace=False))

#First difference with seasonality removed:
ddf['seasonality_first_difference'] = ddf.first_difference - ddf.first_difference.shift(12)
test_stationarity(ddf.seasonality_first_difference.dropna(inplace=False))

#Plot ACF and PACF charts
fig = sm.graphics.tsa.plot_acf(ddf.seasonality_first_difference.iloc[13:], lags=40)
fig = sm.graphics.tsa.plot_pacf(ddf.seasonality_first_difference.iloc[13:], lags=40)
#Create SARIMA Model:
#ARIMA(1, 0, 1)x(1, 1, 1, 12)12 - AIC
mod = sm.tsa.statespace.SARIMAX(ddf.MWh, trend='n', order=(1,0,1), seasonal_order=(1,1,1,12))
results = mod.fit()
print(results.summary())


#Forecasting:
ddf['forecast'] = results.predict(start = 2000, end= 2191)  
ddf[['MWh', 'forecast']].plot(figsize=(12, 8))

#Visualize actual results:
npredict =ddf.MWh['2018'].shape[0]
fig, ax = plt.subplots(figsize=(12,6))
npre = 191
ax.set(title='MWh Real Time System Demand', xlabel='Date', ylabel='MWh')
ax.plot(ddf.index[-npredict-npre+1:], ddf.ix[-npredict-npre+1:, 'MWh'], 'o', label='Observed')
ax.plot(ddf.index[-npredict-npre+1:], ddf.ix[-npredict-npre+1:, 'forecast'], 'g', label='Dynamic forecast')
legend = ax.legend(loc='lower right')
legend.get_frame().set_facecolor('w')

#Future Forecast
start = datetime.datetime.strptime("2018-12-01", "%Y-%m-%d")
date_list = [start + relativedelta(months=x) for x in range(0,12)]
future = pd.DataFrame(index=date_list, columns= ddf.columns)
ddf = pd.concat([ddf, future])


future_forecast = results.predict(n_periods=2191)
ddf['forecast'] = future_forecast

ddf['forecast'] = results.predict(start = 2192, end = 2291, dynamic= True)
ddf[['MWh', 'forecast']].ix[2150:2250].plot(figsize=(12, 8)) 

