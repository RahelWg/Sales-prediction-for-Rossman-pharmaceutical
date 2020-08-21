# -*- coding: utf-8 -*-%matplotlib inline

import numpy as np
import pandas as pd
from pandas import datetime

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import statsmodels.api as sm
import scipy.stats as stats
from statsmodels.distributions.empirical_distribution import ECDF
from sklearn.preprocessing import StandardScaler

# time series analysis
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

import pickle
# prophet by Facebook
#from fbprophet import Prophet

#import dataset
train = pd.read_csv('train.csv', parse_dates = True, low_memory = False, index_col = 'Date' )
test = pd.read_csv('test.csv')
store = pd.read_csv('store.csv')
train_all = pd.read_csv('train.csv', parse_dates = True, low_memory = False, index_col = 'Date' )

# data extraction
train['Year'] = train.index.year
train['Month'] = train.index.month
train['Day'] = train.index.day
train['WeekOfYear'] = train.index.weekofyear

# adding new variable
train['SalePerCustomer'] = train['Sales']/train['Customers']

# closed stores
closed_stores = train[(train.Open == 0) & (train.Sales == 0)]

# opened stores with zero sales
zero_sales = train[(train.Open != 0) & (train.Sales == 0)]

#Removing closed stores
train = train[(train["Open"] != 0) & (train['Sales'] != 0)]

# missing values in CompetitionDistance
# fill NaN with a median value (skewed distribuion)
# since no particular pattern observed in the missing values
store['CompetitionDistance'].fillna(store['CompetitionDistance'].median(), inplace = True)

# replace NA's by 0
store.fillna(0, inplace = True)

# by specifying inner join we make sure that only those observations 
# that are present in both train and store sets are merged together
train_store = pd.merge(train, store, how = 'inner', on = 'Store')

# store types
train_store.groupby('StoreType')['Sales']



# importing data
df = pd.read_csv('train.csv',  low_memory = False)

# remove closed stores and those with no sales
df = df[(df["Open"] != 0) & (df['Sales'] != 0)]

# sales for the store number 1 (StoreType C)
sales = df[df.Store == 1].loc[:, ['Date', 'Sales']]

# reverse to the order: from 2013 to 2015
sales = sales.sort_index(ascending = False)

# to datetime64
sales['Date'] = pd.DatetimeIndex(sales['Date'])


# from the prophet documentation every variables should have specific names
sales = sales.rename(columns = {'Date': 'ds',
                                'Sales': 'y'})


# create holidays dataframe
state_dates = df[(df.StateHoliday == 'a') | (df.StateHoliday == 'b') & (df.StateHoliday == 'c')].loc[:, 'Date'].values
school_dates = df[df.SchoolHoliday == 1].loc[:, 'Date'].values

state = pd.DataFrame({'holiday': 'state_holiday',
                      'ds': pd.to_datetime(state_dates)})
school = pd.DataFrame({'holiday': 'school_holiday',
                      'ds': pd.to_datetime(school_dates)})

holidays = pd.concat((state, school))      


# set the uncertainty interval to 95% (the Prophet default is 80%)
my_model = Prophet(interval_width = 0.95, 
                   holidays = holidays)
my_model.fit(sales)

# dataframe that extends into future 6 weeks 
future_dates = my_model.make_future_dataframe(periods = 6*7)


# predictions
forecast = my_model.predict(future_dates)

# preditions for last week
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]


fc = forecast[['ds', 'yhat']].rename(columns = {'Date': 'ds', 'Forecast': 'yhat'})

# Saving model to disk
pickle.dump(fc, open('rossmann.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('rossmann.pkl','rb'))
#print(model.predict([[2]]))
