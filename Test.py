import pandas as pd
import yfinance as yf
from yahoofinancials import YahooFinancials
import matplotlib.pyplot as plt

ticker = yf.Ticker('MSFT')

tsla_df = ticker.history(period="max")

# plt.plot(tsla_df['Close'], title="TSLA's stock price")
# plt.show()

# print(ticker.info)
# print(ticker.calendar)
print(ticker.recommendations)