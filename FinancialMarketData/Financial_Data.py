import yfinance as yf
import pandas as pd

stocks = ["AAPL", "GOOGL", "MSFT"]
start_date = "2023-01-01"
end_date = "2025-01-01"

data = yf.download(stocks, start=start_date, end=end_date)
data.to_csv("financial_market_data.csv")
print("Stock market data saved successfully.")
