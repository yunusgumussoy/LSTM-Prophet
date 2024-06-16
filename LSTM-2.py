# -*- coding: utf-8 -*-
"""
Created on Thu May 30 12:34:04 2024

@author: Yunus
"""

"""
This code is a machine learning script that trains and evaluates multiple LSTM models with different hyperparameter combinations for predicting the S&P 500 index.
It takes about 15 hours, and here is the results:

     Lag    Nöron  Epochs  Nöron2   TrainRMSE    TestRMSE
407  17.0  128.0   200.0    64.0   21.275000   61.808739
719  30.0  128.0   200.0    64.0   22.291265   62.643084
455  19.0  128.0   200.0    64.0   23.122371   66.905557
551  23.0  128.0   200.0    64.0   23.491694   67.805168
501  21.0  128.0   150.0    64.0   24.112060   69.919515
"""
# pip install --upgrade keras
# pip install tensorflow

import os
os.environ["TF_ENABLE_ONEDNN_OPTNS"]="0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]="1"

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf

from sklearn.preprocessing import MinMaxScaler
from keras.api.models import Sequential
from keras.api.layers import LSTM, Dense, Dropout
from keras.api.optimizers import Adam
from sklearn.metrics import mean_squared_error 
from keras.api.callbacks import EarlyStopping, ModelCheckpoint

endeks=pd.DataFrame(yf.download("^SPX", period="20Y")["Adj Close"])

sc=MinMaxScaler(feature_range=(0,1))
veri_sc=sc.fit_transform(endeks)

train_size=int(len(veri_sc)*0.70)
train,test=veri_sc[0:train_size], veri_sc[train_size:]

def ts (data,timestep):
    x,y=[],[]
    for i in range (timestep, len(data)):
        x.append(data[i-timestep:i,0])
        y.append(data[i,0])
    return np.array(x), np.array(y)

df=pd.DataFrame(columns=["Lag", "Nöron", "Epochs", "Nöron2", "TrainRMSE", "TestRMSE"])

lag=range(1,31)
units=[32,64,128]
units2=[32,64]
epochs=[50,100,150,200]



for i in lag:
    for j in units:
        for k in epochs:
            for u in units2:
            
                x_train, y_train = ts(train, i)
                x_test, y_test = ts(test, i)

                x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
                x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
                y_train = y_train.reshape(-1, 1)
                y_test = y_test.reshape(-1, 1)

                model = Sequential()
                model.add(LSTM(units=j, input_shape=(x_train.shape[1], x_train.shape[2]), return_sequences=True))
                model.add(Dropout(0.2))
                model.add(LSTM(units=u))
                model.add(Dropout(0.2))
                model.add(Dense(y_train.shape[1]))

                opt = Adam(learning_rate=0.0001)

                model.compile(optimizer=opt, loss="mse")

                early_stop = EarlyStopping(monitor='val_loss', patience=10)
                best_model = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True)

                model.fit(x_train, y_train, epochs=k, validation_data=(x_test, y_test), callbacks=[early_stop, best_model])

                model.load_weights('best_model.keras')

                traintahmin = model.predict(x_train)
                testtahmin = model.predict(x_test)

                traintahmin = sc.inverse_transform(traintahmin)
                testtahmin = sc.inverse_transform(testtahmin)
                trainY = sc.inverse_transform(y_train)
                testY = sc.inverse_transform(y_test)

                rmse_train = np.sqrt(mean_squared_error(trainY, traintahmin))
                rmse_test = np.sqrt(mean_squared_error(testY, testtahmin))

                df = df._append({"Lag": i, "Nöron": j, "Epochs": k, "Nöron2": u, "TrainRMSE": rmse_train, "TestRMSE": rmse_test}, ignore_index=True)

df=df.sort_values(by="TrainRMSE")
print(df)

"""
fig,axs=plt.subplots(2,1)

axs[0].plot(traintahmin, label="Train Tahmin")
axs[0].plot(trainY, label="TrainY")
axs[0].legend()

axs[1].plot(testtahmin, label="Test Tahmin")
axs[1].plot(testY, label="TestY")
axs[1].legend()

plt.show()

"""
