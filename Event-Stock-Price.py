# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 02:05:58 2024

@author: Yunus

Source Code: https://medium.com/@urazakgul/filistin-i̇srail-savaşının-thyao-hisse-senedine-etkisinin-event-study-ile-i̇ncelenmesi-465e78f41047
"""

import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import shapiro, levene, ttest_rel
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

def plot_stock_data(tickers, start_date, end_date, title):

    df = pd.DataFrame({'Date': pd.date_range(start=start_date, end=end_date)})

    for ticker in tickers:
        data = yf.download(ticker, start=start_date, end=end_date)
        data = data[['Adj Close']]
        data = data.rename(columns={'Adj Close': ticker})
        df = df.merge(data, left_on='Date', right_index=True, how='left')

    for ticker in tickers:
        df[ticker] = (df[ticker] - df[ticker].min()) / (df[ticker].max() - df[ticker].min())

    df = df.dropna()

    plt.figure(figsize=(12, 6))
    for ticker in tickers:
        plt.plot(df['Date'], df[ticker], label=ticker)

    plt.axvline(x=pd.to_datetime('2023-10-07'), color='black', linestyle='--', label='Oct 7, 2023')
    plt.title(title)
    plt.ylabel('Normalized Closing Prices')
    plt.legend()
    plt.text(
        0.8,
        -0.15,
        'Data source: Yahoo Finance',
        transform=plt.gca().transAxes,
        fontsize=10,
        style='italic',
        color='gray'
    )
    plt.show()

tickers1 = ['^TA125.TA', 'XU100.IS']
start_date1 = '2023-08-28'
end_date1 = '2023-10-20'

title1 = 'TA125 and BIST100 Over the Last 8 Weeks'
plot_stock_data(tickers1, start_date1, end_date1, title1)

tickers2 = ['THYAO.IS', 'XU100.IS']
start_date2 = '2023-08-28'
end_date2 = '2023-10-21'

title2 = 'THYAO and BIST100 Over the Last 8 Weeks'
plot_stock_data(tickers2, start_date2, end_date2, title2)

### Event Study ###

def calculate_event_window(ticker, estimation_start_date, event_date, event_end_date, pre_event_days, post_event_days, alpha=0.05):
    event_df = yf.download(
        tickers=ticker,
        start=estimation_start_date,
        end=event_end_date
    )['Adj Close']

    event_df = pd.DataFrame(event_df).reset_index()
    event_df['Log_Return'] = np.log(event_df['Adj Close'] / event_df['Adj Close'].shift(1))
    event_df = event_df.dropna().reset_index(drop=True)

    event_date_index = event_df[event_df['Date'] == event_date].index[0]
    event_df['Day'] = 0

    for i in range(1, event_date_index + 1):
        event_df.loc[event_date_index - i, 'Day'] = -i

    for i in range(1, len(event_df) - event_date_index):
        event_df.loc[event_date_index + i, 'Day'] = i

    event_df['Type'] = event_df['Day'].apply(lambda x: 'Estimation Window' if x < -pre_event_days else ('Event Window' if -pre_event_days <= x <= post_event_days else ''))

    estimation_window_returns = event_df[event_df['Type'] == 'Estimation Window']['Log_Return']
    average_return_estimation_window = estimation_window_returns.mean()

    event_window_df = event_df[event_df['Type'] == 'Event Window']

    event_window_df = event_window_df.copy()

    event_window_df['Abnormal_Return'] = event_window_df['Log_Return'] - average_return_estimation_window

    event_window_df['Cumulative_Abnormal_Return'] = event_window_df['Abnormal_Return'].cumsum()

    event_window_df["Groups"] = None

    event_window_df.loc[event_window_df["Day"] < 0, "Groups"] = "Before Event"
    event_window_df.loc[event_window_df["Day"] > 0, "Groups"] = "After Event"
    event_window_df.loc[event_window_df["Day"] == 0, "Groups"] = "Event Day"

    before_event = event_window_df[event_window_df["Groups"] == "Before Event"]["Cumulative_Abnormal_Return"]
    after_event = event_window_df[event_window_df["Groups"] == "After Event"]["Cumulative_Abnormal_Return"]

    statistic, p_value = shapiro(event_window_df['Cumulative_Abnormal_Return'])

    if p_value >= alpha:
        print("Data follows a normal distribution.")
    else:
        print("Data does not follow a normal distribution.")
        return

    levene_statistic, levene_p_value = levene(before_event, after_event)

    if levene_p_value < alpha:
        print("The variances of 'Before Event' and 'After Event' groups are not homogeneous.")
    else:
        print("The variances of 'Before Event' and 'After Event' groups are homogeneous.")

    t_stat, p_value = ttest_rel(before_event, after_event)

    if p_value < alpha:
        print("There is a significant difference between 'Before Event' and 'After Event' groups.")
    else:
        print("There is no significant difference between 'Before Event' and 'After Event' groups.")

    return event_df, event_window_df

def plot_estimation_window_returns(estimation_window_returns):
    plt.figure(figsize=(12, 6))
    sns.set(style="whitegrid")
    sns.kdeplot(event_df[event_df['Type'] == 'Estimation Window']['Log_Return'], fill=True, label='Estimation Window')
    mean_log_return = event_df[event_df['Type'] == 'Estimation Window']['Log_Return'].mean()
    plt.axvline(mean_log_return, color='blue', linestyle='--', label='Average')
    plt.axvline(0, color='red', linestyle='--')
    plt.title("Estimation Window Log Returns")
    plt.xlabel("Log Returns")
    plt.ylabel("Density")
    plt.legend()
    plt.show()

def plot_estimation_and_all_log_returns(event_df):
    plt.figure(figsize=(12, 6))
    sns.set(style="whitegrid")
    sns.kdeplot(event_df[event_df['Type'] == 'Estimation Window']['Log_Return'], fill=True, label='Estimation Window')
    mean_estimation_window = event_df[event_df['Type'] == 'Estimation Window']['Log_Return'].mean()
    plt.axvline(mean_estimation_window, color='blue', linestyle='--', label='Estimation Window Average')
    plt.axvline(0, color='red', linestyle='--')
    sns.kdeplot(event_df['Log_Return'], fill=True, label='All Data')
    mean_all_data = event_df['Log_Return'].mean()
    plt.axvline(mean_all_data, color='green', linestyle='--', label='All Data Average')
    plt.title("Estimation and All Log Returns")
    plt.xlabel("Log Returns")
    plt.ylabel("Density")
    plt.legend()
    plt.show()

def plot_abnormal_return_over_event_window(event_window_df):
    plt.figure(figsize=(12, 6))
    plt.plot(event_window_df['Day'], event_window_df['Abnormal_Return'], marker='o', linestyle='-', color='b')
    plt.axvline(x=0, color='black', linestyle='--', label='Event Day')
    plt.axhline(0, color='red', linestyle='--', linewidth=1.2)
    plt.title('Abnormal Return Over Event Window')
    plt.ylabel('Abnormal Return')
    plt.xticks(event_window_df['Day'])
    plt.legend()
    plt.show()

def plot_cumulative_abnormal_return(event_window_df):
    plt.figure(figsize=(12, 6))
    plt.plot(event_window_df['Day'], event_window_df['Cumulative_Abnormal_Return'], marker='o', linestyle='-', color='b')
    plt.axvline(x=0, color='black', linestyle='--', label='Event Day')
    plt.axhline(0, color='red', linestyle='--', linewidth=1.2)
    plt.title('Cumulative Abnormal Return Over Event Window')
    plt.ylabel('Abnormal Return')
    plt.xticks(event_window_df['Day'])
    plt.legend()
    plt.show()

ticker = 'THYAO.IS'
estimation_start_date = '2023-05-15'
event_date = '2023-10-09'
event_end_date = '2023-10-21'
pre_event_days = 9
post_event_days = 9

event_df, result_df = calculate_event_window(
    ticker,
    estimation_start_date,
    event_date,
    event_end_date,
    pre_event_days,
    post_event_days
)

plot_estimation_window_returns(event_df)
plot_estimation_and_all_log_returns(event_df)
plot_abnormal_return_over_event_window(result_df)
plot_cumulative_abnormal_return(result_df)