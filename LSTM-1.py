# -*- coding: utf-8 -*-
"""
Created on Thu May 30 12:34:04 2024

@author: Yunus
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

endeks=pd.DataFrame(yf.download("^SPX", period="20Y")["Adj Close"])

# print(endeks) #kontrol
# print(endeks.isnull().sum()) #eksik veri kontrol

# görselleştirme
# plt.plot(endeks)
# plt.show()

# standardize
sc=MinMaxScaler(feature_range=(0,1))
veri_sc=sc.fit_transform(endeks)
#print(veri_sc.shape) #kontrol

# datayı eğitim ve test olarak bölüyoruz
train_size=int(len(veri_sc)*0.70)
train,test=veri_sc[0:train_size], veri_sc[train_size:]
#print(train.shape) #kontrol
#print(test.shape) #kontrol

# lag/gecikme
def ts (data,timestep):
    x,y=[],[]
    for i in range (timestep, len(data)):
        x.append(data[i-timestep:i,0])
        y.append(data[i,0])
    return np.array(x), np.array(y)

# tahminsel olarak 12 değeri verdik, sonra optimize edeceğiz
x_train, y_train=ts(train,12)
x_test,y_test=ts(test,12)

x_train=x_train.reshape(x_train.shape[0], x_train.shape[1],1)
x_test=x_test.reshape(x_test.shape[0], x_test.shape[1],1)
y_train=y_train.reshape(-1,1)
y_test=y_test.reshape(-1,1)

# print(x_train.shape, y_train.shape, x_test.shape, y_test.shape) #kontrol

# Model
model=Sequential()
model.add(LSTM(64, input_shape=(x_train.shape[1], x_train.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(y_train.shape[1]))

opt=Adam(learning_rate=0.0001) # modeli optimize etmeye çalışıyoruz, normalde adam optimizer ının default öğrenme oranı 0.001 idi.

model.compile(optimizer=opt, loss="mse")
history=model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test))

# modelin hata durumunu görselleştiriyoruz
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Test Loss")
plt.legend()
plt.show()

# test loss da gürültü var, bunu Adam optimizerda değişiklik yaparak gidermeye çalışacağız 


