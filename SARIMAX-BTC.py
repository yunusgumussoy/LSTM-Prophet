import yfinance as yf
import statsmodels.api as sm
import matplotlib.pyplot as plt
import datetime

# Download Bitcoin data
df = yf.download("BTC-USD", start="2014-01-01", end=datetime.date.today())
df = df[['Adj Close']]

# Fit the SARIMA model
model = sm.tsa.statespace.SARIMAX(df, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
results = model.fit()

# Forecasting
forecast = results.get_forecast(steps=30)
predicted_mean = forecast.predicted_mean
conf_int = forecast.conf_int()

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(df.index, df, label='Observed')
plt.plot(predicted_mean.index, predicted_mean, label='Forecast')
plt.fill_between(conf_int.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='pink', alpha=0.3)
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Bitcoin Price Forecast with SARIMA')
plt.legend()
plt.show()
