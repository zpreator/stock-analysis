import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import datetime
import os
from pandas.plotting import register_matplotlib_converters
import yfinance as yf
from sklearn.model_selection import train_test_split
yf.pdr_override()
register_matplotlib_converters()
plt.style.use('fivethirtyeight')

# https://nbviewer.jupyter.org/github/srnghn/ml_example_notebooks/blob/master/Predicting%20Wine%20Types%20with%20Neural%20Networks.ipynb
def GetStonkData(ticker_name, start, end):
    #Get the stock quote 
    df = web.data.get_data_yahoo(ticker_name, data_source='yahoo', start=start, end=end) 
    ticker = yf.Ticker(ticker_name)
    rec = pd.DataFrame(ticker.recommendations)
    rec.index = pd.to_datetime(rec.index, format='%m/%d/%Y').strftime('%Y-%m-%d')
    print(df.head())
    print(rec.head())
    tbl = df.merge(rec, left_index=True, right_index=True)

    grade_list = {'(?i)strong buy': 0 , '(?i)buy': 1, '(?i)outperform': 2, '(?i)overweight': 2, '(?i)positive': 2, '(?i)hold': 3, '(?i)neutral': 3, '(?i)equal-weight': 3, '(?i)underperform': 4, '(?i)underweight': 4, '(?i)negative': 4, '(?i)sell': 5, '(?i)strong sell': 5}
    tbl['agg grade'] = tbl['To Grade'].replace(grade_list, regex=True)

    tomorrow = tbl['Close']
    tbl = tbl[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume','agg grade']]
    tbl = tbl.shift(periods=1)
    tbl['Tomorrow'] = tomorrow
    tbl = tbl.dropna()
    #Show the data 
    print(tbl.head())
    tbl.to_csv(ticker_name + '.csv')
    return tbl

def PrepareData(tbl):
    from sklearn.preprocessing import StandardScaler
    X = tbl.drop(['Tomorrow'], axis=1)
    y = tbl['Tomorrow']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(X_train)
    test_scaled = scaler.transform(X_test)
    return train_scaled, y_train

def TrainModel(train_scaled, y_train):
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import accuracy_score
    model = MLPClassifier()
    model.fit(train_scaled, y_train)
    score = accuracy_score(y_train, model.predict(train_scaled))
    print(score)

def Main():
    tbl = GetStonkData('AAPL', start='2012-01-01', end='2020-08-18')
    train_scaled, y_train = PrepareData(tbl)
    TrainModel(train_scaled, y_train)
    
Main()