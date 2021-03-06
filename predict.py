import pandas as pd
import yfinance as yf
import pandas as pd
import pandas_datareader.data as web
import pandas_ta as ta
import http.client, urllib.parse
import json
from datetime import datetime
import time
from datetime import datetime, timedelta
from dateutil.relativedelta import *
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import GRU
import tensorflow as tf
import math
from zipfile import ZipFile
import pickle
import os
import numpy as np
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import joblib 
from sklearn.ensemble import RandomForestRegressor

def get_sentiment_data(ticker_symbol,df):
    dates = df['Date']
    dates = dates.apply(lambda x: datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S').strftime("%Y-%m-%d"))
    sentiment_values = []

    for m in dates:        
        #the request is put in a try catch block to stop the loop from breaking
        try:
            #initiate the connection to the 3rd party api service
            conn = http.client.HTTPSConnection('api.marketaux.com') 
            params = urllib.parse.urlencode({
                'api_token': 'fkkywOjEYioULrZrV9pt21k6pTtRPW5C17FeWNkE',
                'symbols': ticker_symbol,
                'published_on':m
                })

            conn.request('GET', '/v1/entity/stats/aggregation?{}'.format(params))
            res = conn.getresponse()
            data = res.read()
            parsed = json.loads(data)
            sentiment_values.append(parsed['data'][0]['sentiment_avg'])
        except Exception as e:
            return 'error'      
        
    return sentiment_values


def predict(ticker):
    response = {'status':'success'}
    indicators = ['High','Low','Open','Volume','Adj Close','H-L','O-C','5MA',
              '10MA','20MA','7SD','EMA8','EMA21','EMA34','EMA55','RSI_14','Sentiment']
    
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - relativedelta(months=+4)).strftime('%Y-%m-%d')
    try:
        df = web.DataReader(name=ticker, data_source='yahoo', start=start_date, end=end_date)
    except Exception as e:
        return 'Please check ticker symbol'
    
    df['H-L'] = df['High'] - df['Low']
    df['O-C'] = df['Open'] - df['Close']
    df['5MA'] = df['Adj Close'].rolling(window=5).mean()
    df['10MA'] = df['Adj Close'].rolling(window=10).mean()
    df['20MA'] = df['Adj Close'].rolling(window=20).mean()
    df['7SD'] = df['Adj Close'].rolling(window=7).std()
    df["EMA8"] = df['Adj Close'].ewm(span=8).mean()
    df["EMA21"] = df['Adj Close'].ewm(span=21).mean()
    df['EMA34'] = df['Adj Close'].ewm(span=34).mean()
    df['EMA55'] = df['Adj Close'].ewm(span=55).mean()
    df.ta.rsi(close='Close', length=14, append=True)
    df['Close'] = df['Close'].shift(-1)
    df = df.reset_index()
    sentiment_values =get_sentiment_data(ticker,df)
    
    if sentiment_values == 'error':
        response = {'status':'failure','message':'problem with 3rd party api'}
        return response
    
    df['Sentiment'] = sentiment_values
    today_data = df.iloc[-5:-1][indicators]
    response['close_week'] = np.asarray(df.iloc[-8:-1]['Close'],np.float32).reshape(7).tolist()
    response['open'] = str(round(df.iloc[-1]['Open'],2))
    df.dropna(inplace=True)
    X = np.asarray(df[indicators], np.float32)
    Y = np.asarray(df['Close'], np.float32)
    today_data = np.asarray(today_data,np.float32)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    y_train = y_train.reshape(-1,1)
    mms_X = MinMaxScaler()
    mms_y = MinMaxScaler()

    scaler_x = mms_X.fit(X_train)
    scaler_y = mms_y.fit(y_train)

    X_train = scaler_x.transform(X_train)
    y_train = scaler_y.transform(y_train)

    X_test = scaler_x.transform(X_test)
    y_test = y_test.reshape(-1,1)
    y_test = scaler_y.transform(y_test)

    pca = PCA(n_components=4)

    X_train= pca.fit_transform(X_train)
    X_test = pca.transform(X_test)


    model = RandomForestRegressor(n_estimators = 1000, random_state = 42)
    model.fit(X_train, y_train)

    scaled_data = scaler_x.transform(today_data)
    scaled_data = pca.fit_transform(scaled_data)
   
    
    pred = model.predict(scaled_data)
    pred = scaler_y.inverse_transform(pred.reshape(-1,1))
    response['predicted_close'] = str(round(pred[-1].tolist()[0],2))
    
    
    return response

