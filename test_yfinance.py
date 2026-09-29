import yfinance as yf

ticker = yf.Ticker("AAPL")

hist = ticker.history(period="6mo")

print(hist)