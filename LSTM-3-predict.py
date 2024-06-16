# -*- coding: utf-8 -*-
"""
Created on Thu May 30 12:34:04 2024

@author: Yunus
"""
# pip install --upgrade keras
# pip install tensorflow

import os
os.environ["TF_ENABLE_ONEDNN_OPTNS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf

from sklearn.preprocessing import MinMaxScaler
from keras.api.models import Sequential
from keras.api.layers import LSTM, Dense, Dropout
from keras.api.optimizers import Adam
from sklearn.metrics import mean_squared_error

# Fetch Bitcoin data
endeks = pd.DataFrame(yf.download("BTC-USD", period="5y")["Adj Close"])

# Normalize the data
sc = MinMaxScaler(feature_range=(0, 1))
veri_sc = sc.fit_transform(endeks)

# Split into training and testing sets
train_size = int(len(veri_sc) * 0.70)
train, test = veri_sc[:train_size], veri_sc[train_size:]

# Create time series dataset
def ts(data, timestep):
    x, y = [], []
    for i in range(timestep, len(data)):
        x.append(data[i-timestep:i, 0])
        y.append(data[i, 0])
    return np.array(x), np.array(y)

# Prepare the training and test data
x_train, y_train = ts(train, 30)
x_test, y_test = ts(test, 30)

# Reshape data for LSTM input
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

# Build LSTM model
# According to my trials, 2 layer LSTM model is better than 3 layers
model = Sequential()
model.add(LSTM(units=128, input_shape=(x_train.shape[1], x_train.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=64))
model.add(Dropout(0.2))
model.add(Dense(1))

# Compile and train the model
opt = Adam(learning_rate=0.0001)
model.compile(optimizer=opt, loss="mse")
model.fit(x_train, y_train, epochs=200, validation_data=(x_test, y_test), verbose=1)

# Make predictions
traintahmin = model.predict(x_train)
testtahmin = model.predict(x_test)

# Inverse transform predictions and actual values
traintahmin = sc.inverse_transform(traintahmin)
testtahmin = sc.inverse_transform(testtahmin)
trainY = sc.inverse_transform(y_train)
testY = sc.inverse_transform(y_test)

# Plot predictions and actual values
fig, axs = plt.subplots(2, 1, figsize=(12, 8))

axs[0].plot(trainY, label="Actual Train")
axs[0].plot(traintahmin, label="Predicted Train")
axs[0].legend()

axs[1].plot(testY, label="Actual Test")
axs[1].plot(testtahmin, label="Predicted Test")
axs[1].legend()

plt.show()
