# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 15:39:22 2024

@author: Yunus
"""

import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import warnings
import datetime

# Ignore warnings
warnings.filterwarnings("ignore")

# Download Bitcoin data
df = yf.download("BTC-USD", start="2014-01-01", end=datetime.date.today())
df = df[['Adj Close']]

# Fit the ARIMA model
model = ARIMA(df, order=(5, 1, 0))  # order=(p,d,q)
results = model.fit()

# Forecasting
forecast_steps = 30
forecast = results.get_forecast(steps=forecast_steps)
predicted_mean = forecast.predicted_mean
conf_int = forecast.conf_int()

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(df.index, df, label='Observed')
plt.plot(predicted_mean.index, predicted_mean, label='Forecast')
plt.fill_between(conf_int.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='pink', alpha=0.3)
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Bitcoin Price Forecast with ARIMA')
plt.legend()
plt.show()
