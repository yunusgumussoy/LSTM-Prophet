# -*- coding: utf-8 -*-
"""
Created on Thu May 30 12:34:04 2024

@author: Yunus
"""

import pandas as pd
import yfinance as yf
from prophet import Prophet
import matplotlib.pyplot as plt
import datetime

# Fetch Bitcoin data
df = yf.download("BTC-USD", start="2014-01-01", end=datetime.date.today())
df.reset_index(inplace=True)

# Prepare data for Prophet
df = df[['Date', 'Adj Close']]
df.rename(columns={'Date': 'ds', 'Adj Close': 'y'}, inplace=True)

# Initialize and fit the model
model = Prophet()
model.fit(df)

# Create a dataframe to hold predictions
future = model.make_future_dataframe(periods=180)
forecast = model.predict(future)

# Plot the forecast
fig = model.plot(forecast)
plt.show()

# Plot forecast components
fig2 = model.plot_components(forecast)
plt.show()
