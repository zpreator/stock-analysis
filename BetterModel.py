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
    tbl = tbl[tbl['agg grade'].apply(lambda x: isinstance(x, (int, np.int64)))]

    tomorrow = tbl['Close']
    tbl = tbl[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume','agg grade']]
    tbl = tbl.shift(periods=1)
    tbl['Tomorrow'] = tomorrow
    tbl = tbl.dropna()
    #Show the data 
    print(tbl.head())
    tbl.to_csv(ticker_name + '.csv')
    return tbl

def GetStonkData2(ticker_name, start, end):
    tbl = web.data.get_data_yahoo(ticker_name, data_source='yahoo', start=start, end=end)
    tomorrow = tbl['Close']
    tbl = tbl.shift(periods=1)
    tbl['Tomorrow'] = tomorrow
    tbl = tbl.dropna()
    return tbl

def PrepareData(tbl):
    from sklearn.preprocessing import StandardScaler
    X = tbl.drop(['Tomorrow'], axis=1).astype(float)
    y = tbl['Tomorrow'].astype(float)
    X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2, random_state=123)
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(X_train)
    test_scaled = scaler.transform(X_test)
    return train_scaled, test_scaled, y_train, y_test

def PreparePredict(tbl):
    from sklearn.preprocessing import StandardScaler
    X = tbl.drop(['Tomorrow']).astype(float)
    scaler = StandardScaler()
    X = X.values.reshape(1, -1)
    scaler.fit(X)
    test = scaler.transform(X)
    return test

def TrainModel(train_scaled, test_scaled, y_train, y_test):
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import mean_absolute_error
    from math import sqrt
    tree_model = DecisionTreeRegressor()
    rf_model = RandomForestRegressor()
    tree_model.fit(train_scaled, y_train)
    rf_model.fit(train_scaled, y_train)
    
    tree_mse = mean_squared_error(y_train, tree_model.predict(train_scaled))
    tree_mae = mean_absolute_error(y_train, tree_model.predict(train_scaled))
    rf_mse = mean_squared_error(y_train, rf_model.predict(train_scaled))
    rf_mae = mean_absolute_error(y_train, rf_model.predict(train_scaled))
    print("Decision Tree training mse = ",tree_mse," & mae = ",tree_mae," & rmse = ", sqrt(tree_mse))
    print("Random Forest training mse = ",rf_mse," & mae = ",rf_mae," & rmse = ", sqrt(rf_mse))

    tree_test_mse = mean_squared_error(y_test, tree_model.predict(test_scaled))
    tree_test_mae = mean_absolute_error(y_test, tree_model.predict(test_scaled))
    rf_test_mse = mean_squared_error(y_test, rf_model.predict(test_scaled))
    rf_test_mae = mean_absolute_error(y_test, rf_model.predict(test_scaled))
    print("Decision Tree test mse = ",tree_test_mse," & mae = ",tree_test_mae," & rmse = ", sqrt(tree_test_mse))
    print("Random Forest test mse = ",rf_test_mse," & mae = ",rf_test_mae," & rmse = ", sqrt(rf_test_mse))
    if tree_test_mse > rf_test_mse:
        return rf_model
    else:
        return tree_model

def TrainNN(train_scaled, test_scaled, y_train, y_test):
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.callbacks import EarlyStopping
    callbacks = EarlyStopping(monitor='loss', min_delta=0.001, verbose=1)#stop early if loss function sufficient
        
    model = Sequential()
    model.add(Dense(128, activation='tanh', input_shape=(train_scaled.shape[1],)))
    model.add(Dense(64, activation='tanh'))
    model.add(Dense(64, activation='tanh'))
    model.add(Dense(32, activation='tanh'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    model.fit(train_scaled, y_train, epochs=1000, callbacks=[callbacks])
    score = model.evaluate(test_scaled, y_test)
    print(score)
    return model

def Main():
    tbl = GetStonkData('AAPL', start='1995-01-01', end='2020-08-20')
    train_scaled, test_scaled, y_train, y_test = PrepareData(tbl)
    # model = TrainModel(train_scaled, test_scaled, y_train, y_test)
    model = TrainNN(train_scaled, test_scaled, y_train, y_test)
    # prediction = train_scaled[-1].reshape(1, -1)
    prediction = PreparePredict(tbl.iloc[-1])
    print(tbl.iloc[-1])
    print('Prediction: ', model.predict(prediction))
    
Main()