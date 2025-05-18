import yfinance as yf
import pandas as pd
import logging
import time

# Set up logging
logging.basicConfig(level=logging.INFO)

def get_stock_price(ticker_sym):
    try:
        stock = yf.Ticker(ticker_sym)
        info = stock.info
        current_price = info.get('currentPrice')
        if current_price:
            logging.info(f"  {ticker_sym:<15}: ₹{current_price:<8.2f}")
        else:
            logging.warning(f"  {ticker_sym:<15}: Price not found.")
    except Exception as e:
        logging.warning(f"  Warn: Error fetching price for {ticker_sym}: {str(e)[:100]}", exc_info=False)

# Test with a few ticker symbols
ticker_syms = ['RVNL.NS', 'RELIANCE.NS']
for ticker_sym in ticker_syms:
    get_stock_price(ticker_sym)
    time.sleep(0.1)  # add a small delay between requests