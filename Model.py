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
yf.pdr_override()
register_matplotlib_converters()
plt.style.use('fivethirtyeight')

def GetStonkData(ticker, start, end):
    #Get the stock quote 
    df = web.data.get_data_yahoo(ticker, data_source='yahoo', start=start, end=end) 
    ticker = yf.Ticker(ticker)
    rec = pd.DataFrame(ticker.recommendations)
    rec.index = pd.to_datetime(rec.index, format='%m/%d/%Y').strftime('%Y-%m-%d')
    print(df.head())
    print(rec.head())
    tbl = df.merge(rec, left_index=True, right_index=True)

    grade_list = {'(?i)strong buy': 0 , '(?i)buy': 1, '(?i)outperform': 2, '(?i)overweight': 2, '(?i)positive': 2, '(?i)hold': 3, '(?i)neutral': 3, '(?i)equal-weight': 3, '(?i)underperform': 4, '(?i)underweight': 4, '(?i)negative': 4, '(?i)sell': 5, '(?i)strong sell': 5}
    tbl['agg grade'] = tbl['To Grade'].replace(grade_list, regex=True)

    tbl = tbl['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'agg grade']
    #Show the data 
    print(tbl.head())
    return tbl

def DisplayStonks(df):
    #Visualize the closing price history
    plt.figure(figsize=(16,8))
    plt.title('Close Price History')
    plt.plot(df['Close'])
    plt.xlabel('Date',fontsize=18)
    plt.ylabel('Close Price USD ($)',fontsize=18)
    plt.show()

def PrepareData(df):
    #Create a new dataframe with only the 'Close' column
    data = df.filter(['Close'])
    #Converting the dataframe to a numpy array
    dataset = data.values
    #Get /Compute the number of rows to train the model on
    training_data_len = math.ceil( len(dataset) *.8)
    #Scale the all of the data to be values between 0 and 1 
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    scaled_data = scaler.fit_transform(dataset)

    #Create the scaled training data set 
    train_data = scaled_data[0:training_data_len  , : ]
    #Split the data into x_train and y_train data sets
    x_train=[]
    y_train = []
    for i in range(60,len(train_data)):
        x_train.append(train_data[i-60:i,0])
        y_train.append(train_data[i,0])

    #Convert x_train and y_train to numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)

    #Reshape the data into the shape accepted by the LSTM
    x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

    #Test data set
    test_data = scaled_data[training_data_len - 60: , : ]
    #Create the x_test and y_test data sets
    x_test = []
    y_test =  dataset[training_data_len : , : ] #Get all of the rows from index 1603 to the rest and all of the columns (in this case it's only column 'Close'), so 2003 - 1603 = 400 rows of data
    for i in range(60,len(test_data)):
        x_test.append(test_data[i-60:i,0])

    #Convert x_test to a numpy array 
    x_test = np.array(x_test)
    #Reshape the data into the shape accepted by the LSTM
    x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))

    return x_train, y_train, x_test, y_test, data, scaler

def BuildModel(x_train, y_train):
    from keras.models import Sequential
    from keras.layers import Dense, LSTM
    #Build the LSTM network model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True,input_shape=(x_train.shape[1],1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    #Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    #Train the model
    model.fit(x_train, y_train, batch_size=1, epochs=1)
    return model

def PredictFuture(model, scaler, x_test, y_test, data):
    #Getting the models predicted price values
    predictions = model.predict(x_test)
    #Scale the all of the data to be values between 0 and 1 
    # scaler = MinMaxScaler(feature_range=(0, 1)) 
    predictions = scaler.inverse_transform(predictions)#Undo scaling
    #Calculate/Get the value of RMSE
    rmse=np.sqrt(np.mean(((predictions- y_test)**2)))
    print("Error: ", rmse)

    #Converting the dataframe to a numpy array
    dataset = data.values
    #Get /Compute the number of rows to train the model on
    training_data_len = math.ceil( len(dataset) *.8)

    #Plot/Create the data for the graph
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predictions
    #Visualize the data
    plt.figure(figsize=(16,8))
    plt.title('Model')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
    plt.show()

def PredictStonks(ticker, model_path, date):
    from keras.models import load_model
    model = load_model(model_path)
    #Get the quote
    quote = web.DataReader(ticker, data_source='yahoo', start='2012-01-01', end=date)
    #Create a new dataframe
    new_df = quote.filter(['Close'])
    #Get teh last 60 day closing price 
    last_60_days = new_df[-60:].values

    #Create a new dataframe with only the 'Close' column
    data = new_df.filter(['Close'])
    #Converting the dataframe to a numpy array
    dataset = data.values
    #Get /Compute the number of rows to train the model on
    training_data_len = math.ceil( len(dataset) *.8)
    #Scale the all of the data to be values between 0 and 1 
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    scaled_data = scaler.fit_transform(dataset)
    #Scale the data to be values between 0 and 1
    last_60_days_scaled = scaler.transform(last_60_days)
    #Create an empty list
    X_test = []
    #Append teh past 60 days
    X_test.append(last_60_days_scaled)
    #Convert the X_test data set to a numpy array
    X_test = np.array(X_test)
    #Reshape the data
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    #Get the predicted scaled price
    pred_price = model.predict(X_test)
    #undo the scaling 
    pred_price = scaler.inverse_transform(pred_price)
    print(ticker, ' predicted price on 2020-08-19', pred_price)
    return pred_price

def TrainStonksModel(ticker, save_path, start='2012-01-01', end='2020-08-18'):
    df = GetStonkData(ticker, start, end)
    DisplayStonks(df)
    x_train, y_train, x_test, y_test, data, scaler = PrepareData(df)
    model = BuildModel(x_train, y_train)
    PredictFuture(model, scaler, x_test, y_test, data)
    model.save(save_path)

GetStonkData('AAPL', start='2012-01-01', end='2020-08-18')