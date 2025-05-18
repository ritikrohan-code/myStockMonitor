# -*- coding: utf-8 -*-
# --- IMPORTS ---
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import schedule
import time
from datetime import datetime
import pytz
import statistics
import numpy as np
from flask import Flask, jsonify
from flask_cors import CORS
import threading
import logging
from stocks import indian_stocks_by_sector_py # Import the stock data from the provided file
import traceback  # For detailed error logging
from datetime import datetime, timedelta # Make sure timedelta is imported
import feedparser # For RSS Feeds
import re # For get_yahoo_ticker (likely already there)

# --- Setup Logging ---
# Configure logging to show timestamp, level, thread name, and message
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
# Suppress yfinance's own verbose logging if desired
yf_logger = logging.getLogger('yfinance')
yf_logger.setLevel(logging.WARNING) # Only show warnings and errors from yfinance

# --- Add News Configuration ---
# Define RSS Feeds (Add more reliable sources)
RSS_FEED_URLS = [
    "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms", # ET Markets
    "https://www.moneycontrol.com/rss/marketreports.xml",
    "https://www.moneycontrol.com/rss/business.xml",
    "https://www.livemint.com/rss/markets",
    "https://www.investing.com/rss/news_285.rss", # Investing.com India News
    "https://www.business-standard.com/rss/markets-106.rss", # Business Standard Markets
    "https://www.thehindubusinessline.com/markets/feeder/default.rss", # BusinessLine Markets
    "https://feeds.feedburner.com/nseindia/latest-news", # NSE Feedburner (Check reliability)
]
NEWS_HEADLINE_LIMIT = 50 # Max news items to store/send
NEWS_FETCH_RECENCY_HOURS = 48 # Only keep news from last X hours
NEWS_FETCH_INTERVAL_MINUTES = 30 # How often to check for new news

# 2. Other Configurations
IST = pytz.timezone('Asia/Kolkata')
MARKET_OPEN_TIME = datetime.strptime("09:15", "%H:%M").time()
MARKET_CLOSE_TIME = datetime.strptime("15:30", "%H:%M").time()
TOP_N_STOCKS = 5
HISTORICAL_PERIOD = "1y"
INTRADAY_INTERVAL = "15m"

# --- Breakout Configuration ---
VOLUME_AVG_PERIOD = 20
VOLUME_SURGE_MULTIPLIER = 1.75 # Volume must be 75% above average
RSI_BREAKOUT_THRESHOLD = 60   # Minimum RSI for breakout consideration
MINIMUM_BREAKOUT_SCORE_THRESHOLD = 50 # Minimum points to trigger a breakout signal
MAX_POSSIBLE_BREAKOUT_POINTS = 100 # Define the theoretical max points for normalization

# --- Global Variables (for state sharing and API) ---
latest_news_headlines = [] # Stores list of recent news dicts
monitored_stocks_tickers = [] # List of tickers for monitoring function
pre_market_analysis_done_today = False
latest_sector_data = {}
latest_top_stocks_data = [] # Stores detailed results for top N stocks
last_analysis_time = None
api_lock = threading.Lock() # To safely update/read shared data
latest_breakout_stocks_data = []

# --- Flask App Setup ---
app = Flask(__name__)
CORS(app) # Allow cross-origin requests for React dev server

# --- Helper Functions ---
# --- NEW News Fetching Functions ---
def fetch_news_headlines(rss_feeds):
    """Fetches headlines, links, source, and published time from RSS feeds."""
    headlines_data = []
    processed_links = set() # To avoid duplicates within a single fetch run
    logging.info(f"Fetching news from {len(rss_feeds)} RSS feeds...")

    for url in rss_feeds:
        try:
            # Use request headers to mimic a browser slightly
            headers = {'User-Agent': 'Mozilla/5.0 (compatible; StockMonitorBot/1.0)'} # Identify bot
            feed = feedparser.parse(url, request_headers=headers, agent=headers['User-Agent']) # Pass agent too
            feed_title = feed.feed.get('title', url) # Use URL as fallback source name
            # logging.debug(f"  Parsing '{feed_title}' - Entries: {len(feed.entries)}") # Debug level

            for entry in feed.entries:
                link = entry.get('link', '')
                if not link or link in processed_links: # Skip if no link or already seen
                    continue

                processed_links.add(link)
                # Strip potential HTML tags from title using regex
                title = re.sub('<[^<]+?>', '', entry.get('title', 'No Title')).strip()

                # Attempt to parse published date robustly
                published_parsed = entry.get('published_parsed') or entry.get('updated_parsed')
                published_dt = None
                if published_parsed:
                    try:
                        # Convert time.struct_time to datetime object, make it timezone-aware (UTC initially)
                        utc_dt = datetime.fromtimestamp(time.mktime(published_parsed), tz=pytz.utc)
                        # Convert to IST
                        published_dt = utc_dt.astimezone(IST)
                    except Exception as date_err:
                         # logging.debug(f"    Could not parse date for '{title}': {date_err}")
                         published_dt = datetime.now(IST) # Fallback to current time if parsing fails

                headlines_data.append({
                    'title': title,
                    'link': link,
                    'source': feed_title, # Add source info
                    'published_iso': published_dt.isoformat() if published_dt else None, # Store as ISO string
                    'published_ts': published_dt.timestamp() if published_dt else 0 # Store timestamp for sorting
                })
            time.sleep(0.2) # Be polite between feeds
        except Exception as e:
            logging.warning(f"Failed to parse feed {url}: {e}", exc_info=False)

    logging.info(f"Fetched {len(headlines_data)} unique headlines this run.")
    return headlines_data

def fetch_and_store_latest_news():
    """Fetches news, filters, sorts, limits, and updates global state."""
    global latest_news_headlines # Declare global to modify it
    logging.info("--- Running News Fetch Task ---")
    try:
        # Fetch new headlines using the defined URLs
        new_headlines = fetch_news_headlines(RSS_FEED_URLS)
        if not new_headlines:
            logging.info("No new headlines fetched in this run.")
            return # Nothing to update

        # Get existing headlines safely
        with api_lock:
            current_headlines = list(latest_news_headlines)

        # Combine and remove duplicates based on link
        # Use a dictionary for efficient duplicate removal (keeps latest entry for a given link)
        combined_headlines_dict = {h['link']: h for h in current_headlines}
        for h in new_headlines:
             combined_headlines_dict[h['link']] = h # New entries overwrite old if link is same

        combined_headlines = list(combined_headlines_dict.values())

        # Filter by recency using timestamp comparison
        cutoff_timestamp = (datetime.now(IST) - timedelta(hours=NEWS_FETCH_RECENCY_HOURS)).timestamp()
        recent_headlines = [h for h in combined_headlines if h.get('published_ts', 0) >= cutoff_timestamp]

        # Sort by published time (newest first) using timestamp
        recent_headlines.sort(key=lambda x: x.get('published_ts', 0), reverse=True)

        # Limit the number of headlines stored
        limited_headlines = recent_headlines[:NEWS_HEADLINE_LIMIT]

        # Update global state safely
        with api_lock:
            latest_news_headlines = limited_headlines
            logging.info(f"Updated global news state. Stored {len(latest_news_headlines)} headlines.")

    except Exception as e:
        logging.error(f"Error during fetch_and_store_latest_news: {e}", exc_info=True)

def safe_json_convert(value):
    """
    Converts NumPy numeric/boolean types and handles NaN/Inf for JSON,
    compatible with NumPy >= 2.0.
    """
    # 1. Handle NaN/NaT/None first
    if pd.isna(value):
        return None

    # 2. Check against NumPy abstract base classes for numbers
    if isinstance(value, np.integer): # Catches np.int8, np.int16, np.int32, np.int64, etc.
        return int(value)
    elif isinstance(value, np.floating): # Catches np.float16, np.float32, np.float64, etc.
        # Handle Infinity separately as JSON doesn't support it
        if np.isinf(value):
            # Represent infinity as None or a large number string
            # Using None is generally safer for frontend handling
            return None
            # return str(value) # Alternative: return '+inf' or '-inf' as string
        return float(value)
    # 3. Check for NumPy boolean (np.bool_ is the scalar type)
    elif isinstance(value, np.bool_):
        return bool(value)
    # 4. Handle potential void types from structured arrays (less common here)
    elif isinstance(value, (np.void)):
        return None
    # 5. Assume standard Python types are already JSON serializable
    # (Handles str, list, dict, standard int, float, bool, None directly)
    return value

# --- API Endpoints ---
@app.route('/api/status', methods=['GET'])
def get_status():
    """Returns the current status of the backend analysis."""
    with api_lock:
        # Access global variables safely
        status = {
            'pre_market_analysis_done_today': pre_market_analysis_done_today,
            'last_analysis_time': last_analysis_time.isoformat() if last_analysis_time else None,
            'monitoring_active': bool(monitored_stocks_tickers),
            'monitored_stock_count': len(monitored_stocks_tickers)
        }
    return jsonify(status)

@app.route('/api/sector-performance', methods=['GET'])
def get_sectors():
    """Returns the latest calculated sector performance data."""
    with api_lock:
        # Return a copy to avoid modification issues if any
        data_to_return = dict(latest_sector_data)
    # Convert dict to list of objects for easier frontend mapping
    sector_list = [{"name": name, **info} for name, info in data_to_return.items()]
    return jsonify(sector_list)

@app.route('/api/top-stocks', methods=['GET'])
def get_top_stocks():
    """Returns the detailed analysis results for the latest top N stocks."""
    with api_lock:
        # Return a copy
        data_to_return = list(latest_top_stocks_data)
    return jsonify(data_to_return)

# --- NEW News API Endpoint ---
@app.route('/api/latest-news', methods=['GET'])
def get_latest_news():
    """Returns the latest fetched news headlines."""
    with api_lock:
        # Return a copy of the current news list
        data_to_return = list(latest_news_headlines)
    return jsonify(data_to_return)


@app.route('/api/breakout-stocks', methods=['GET'])
def get_breakout_stocks():
    """Returns the detailed analysis results for stocks with strong breakout signals."""
    with api_lock:
        # Return a copy
        data_to_return = list(latest_breakout_stocks_data)
    
    logging.info(f"Returning {len(data_to_return)} breakout stocks.")
    return jsonify(data_to_return)

# --- Core Functions ---

def get_yahoo_ticker(symbol):
    """Consistent mapping from symbol to Yahoo Finance ticker."""
    symbol_upper = symbol.upper() # Use uppercase for comparisons
    if symbol_upper == "M&M": return "M&M.NS"
    if symbol_upper == "VODAFONEIDEA": return "IDEA.NS"
    if symbol_upper == "IIFL": return "IIFL.NS" # Assuming IIFL Finance, verify if needed
    # Add other specific overrides here if needed
    # e.g., if 'XYZ-EQ' should become 'XYZ.NS'
    if '-' in symbol_upper: return symbol_upper + ".NS"
    return symbol_upper + ".NS"

def get_sector_performance():
    """
    Calculates average performance of sectors based on constituent stocks.
    Also identifies stocks with strong breakout conditions.
    """
    global latest_breakout_stocks_data # Modify the global list
    logging.info("Calculating sector performance and identifying breakout stocks...")
    sector_avg_performance = {}
    total_stocks_to_process = sum(len(v) for v in indian_stocks_by_sector_py.values())
    processed_count = 0
    STRONG_BREAKOUT_CONFIDENCE_THRESHOLD = 75 # Local var

    temp_breakout_stocks = [] # Local accumulation

    for sector_name, stock_list in indian_stocks_by_sector_py.items():
        sector_stock_performances = []
        for stock_info in stock_list:
            processed_count += 1
            if processed_count % 10 == 0:
                logging.info(f"  Sector Perf Progress: {processed_count}/{total_stocks_to_process}")

            ticker_ns = get_yahoo_ticker(stock_info['symbol'])
            try:
                stock = yf.Ticker(ticker_ns)
                hist = stock.history(period="3d", auto_adjust=False, repair=True)
                if hist.empty or len(hist) < 2: continue

                last_close = hist['Close'].iloc[-1]
                prev_close = hist['Close'].iloc[-2]

                if pd.notna(last_close) and pd.notna(prev_close) and prev_close != 0:
                    performance = ((last_close - prev_close) / prev_close) * 100
                    sector_stock_performances.append(performance)


                    # --- BREAKOUT CHECK INTEGRATION ---
                    # Analyze stock data for breakouts here
                    analysis_result = analyze_stock(ticker_ns) # Re-uses your function!
                    if analysis_result and analysis_result.get('breakout_signal'):
                         confidence = analysis_result.get('breakout_confidence', 0)
                         if confidence >= STRONG_BREAKOUT_CONFIDENCE_THRESHOLD:
                              temp_breakout_stocks.append(analysis_result)
                              logging.info(f"    BREAKOUT: {ticker_ns} Confidence={confidence}%")

            except Exception as e:
                logging.warning(f"    Warn: Could not fetch/process {ticker_ns} for sector perf. Error: {str(e)[:100]}", exc_info=False)
                pass
            time.sleep(0.05)

        if sector_stock_performances:
            try:
                average_perf = statistics.mean(sector_stock_performances)
                sector_avg_performance[sector_name] = {
                    'average_performance': average_perf,
                    'stocks_calculated': len(sector_stock_performances),
                    'stocks_defined': len(stock_list)
                }
            except statistics.StatisticsError:
                logging.warning(f"  Could not calculate mean for sector {sector_name} (empty list after errors?).")

    if not sector_avg_performance:
        logging.warning("No sector performance could be calculated.")
        return {}

    logging.info("Sorting sectors by calculated average performance...")
    sorted_sectors = dict(sorted(
        sector_avg_performance.items(),
        key=lambda item: item[1].get('average_performance', -float('inf')),
        reverse=True
    ))

    # Update Global AFTER ALL Processing is Completed.
    with api_lock:
        latest_breakout_stocks_data = temp_breakout_stocks # atomic update.
    

    logging.info(f"  Identified {len(latest_breakout_stocks_data)} strong breakout stocks during sector analysis.")

    return sorted_sectors

# --- NEW: Function to run periodic sector performance check ---
def run_periodic_sector_update():
    """Checks market time and runs sector performance update if within hours."""
    global latest_sector_data, api_lock # Need access to global state and lock

    now_ist = datetime.now(IST)
    current_time = now_ist.time()

    # Check if current time is within market hours
    if MARKET_OPEN_TIME <= current_time < MARKET_CLOSE_TIME:
        logging.info("--- Running Periodic Sector Performance Update (Market Open) ---")
        try:
            # Call the existing function to get updated performance
            updated_sector_data = get_sector_performance()

            if updated_sector_data: # Ensure data was returned
                # Update the global state safely for the API endpoint
                with api_lock:
                    latest_sector_data = updated_sector_data
                logging.info("Periodic sector performance state updated successfully.")
            else:
                 logging.warning("Periodic sector update ran, but get_sector_performance returned no data.")

        except Exception as e:
            logging.error(f"Error during periodic sector performance update: {e}", exc_info=True)
            # Optionally revert to old data or keep it as is on error?
            # Current implementation keeps the last successful data.
    else:
        # This log can be noisy, maybe set to DEBUG level if too frequent
        logging.debug("Skipping periodic sector update (Market Closed).")

def check_breakout_conditions(df):
    """ Analyzes DataFrame for complex breakout conditions including Volume Surge and BBands. """
    is_breakout = False
    achieved_score = 0
    reasons = []
    additional_info = {}  # Store extra info for debugging/display

    # --- Scoring Constants ---
    POINTS_PRICE_VS_SMA50 = 15
    POINTS_PRICE_VS_BBAND = 25
    POINTS_VOLUME_SURGE = 30
    POINTS_RSI = 15
    POINTS_MACD_CROSS = 15
    POINTS_CHART_PATTERN = 15 # Add points for chart pattern

    # Total points should match MAX_POSSIBLE_BREAKOUT_POINTS if all are met = 100

    min_rows_needed = max(51, VOLUME_AVG_PERIOD + 1)
    if df is None or df.empty or len(df) < min_rows_needed:
        return is_breakout, 0, ["Insufficient historical data"], additional_info  # Return additional_info too

    required_cols = ['Close', 'Volume', 'SMA_50', 'RSI_14',
                     'MACD_12_26_9', 'MACDs_12_26_9',
                     'BBU_20_2.0', f'SMA_{VOLUME_AVG_PERIOD}_VOL']

    # Check if all required columns exist and their *last* value is not NaN
    if not all(col in df.columns and pd.notna(df[col].iloc[-1]) for col in required_cols):
        missing_or_nan = [
            col for col in required_cols if col not in df.columns or pd.isna(df[col].iloc[-1])
        ]
        return is_breakout, 0, [f"Missing/NaN indicators: {', '.join(missing_or_nan)}"], additional_info

    try:
        # Get latest values (use .iloc[-1] for the most recent row)
        last_row = df.iloc[-1]
        last_close = last_row['Close']
        last_vol = last_row['Volume']
        sma50 = last_row['SMA_50']
        rsi = last_row['RSI_14']
        macd_line = last_row['MACD_12_26_9']
        macd_signal = last_row['MACDs_12_26_9']
        upper_bb = last_row['BBU_20_2.0']
        avg_vol = last_row[f'SMA_{VOLUME_AVG_PERIOD}_VOL']

        # --- Evaluate Conditions ---
        # Use concise reason strings for better display
        if last_close > sma50:
            achieved_score += POINTS_PRICE_VS_SMA50
            reasons.append(f"Price>SMA50")
        if last_close > upper_bb:
            achieved_score += POINTS_PRICE_VS_BBAND
            reasons.append(f"Price>UpperBB")
        # Ensure avg_vol is not zero or NaN before dividing
        if pd.notna(avg_vol) and avg_vol > 0 and last_vol > (avg_vol * VOLUME_SURGE_MULTIPLIER):
             achieved_score += POINTS_VOLUME_SURGE
             vol_ratio = last_vol / avg_vol
             reasons.append(f"VolSurge({vol_ratio:.1f}x)")
             additional_info['volume_surge_ratio'] = round(vol_ratio, 1)
        if rsi > RSI_BREAKOUT_THRESHOLD:
            achieved_score += POINTS_RSI
            reasons.append(f"RSI>{RSI_BREAKOUT_THRESHOLD}({rsi:.0f})")
        if macd_line > macd_signal:
            # Optional: Check if cross happened recently (e.g., MACD was below signal yesterday)
            # if len(df) > 1 and df['MACD_12_26_9'].iloc[-2] <= df['MACDs_12_26_9'].iloc[-2]:
            achieved_score += POINTS_MACD_CROSS
            reasons.append(f"MACD>Signal")
            # Store more details of the MACD for API to display
            additional_info['macd_value'] = round(macd_line, 2)
            additional_info['macd_signal_value'] = round(macd_signal, 2)

        # --- Chart Pattern Recognition (Simple Example) ---
        # This is a *very* basic example.  Real chart pattern recognition is complex and often involves image processing techniques.  This assumes you're pre-calculating some pattern metric externally.
        # Look for higher highs and higher lows
        last_5_highs = df['Close'].tail(5)
        last_5_lows = df['Close'].tail(5)

        # Check for a general upward trend
        if all(last_5_highs.diff().dropna() > 0) and all(last_5_lows.diff().dropna() > 0):
            achieved_score += POINTS_CHART_PATTERN
            reasons.append("Uptrend Pattern")
            additional_info['chart_pattern'] = "Uptrend"
        else:
            additional_info['chart_pattern'] = "None"

        # --- Determine Final Breakout Status and Confidence ---
        confidence_score = 0
        if achieved_score >= MINIMUM_BREAKOUT_SCORE_THRESHOLD:
            is_breakout = True
            # Normalize score to 0-100 range based on points achieved
            confidence_score = min(100, int(round((achieved_score / MAX_POSSIBLE_BREAKOUT_POINTS) * 100)))
        # Keep reasons even if below threshold for potential debugging/info
        # if not is_breakout: reasons = ["Score below threshold"] if not reasons else reasons


    except KeyError as e:
        logging.warning(f"Warn: Breakout check missing column '{e}'", exc_info=False)
        return is_breakout, 0, [f"Missing col '{e}'"], additional_info
    except Exception as e:
        logging.warning(f"Warn: Error during breakout condition check: {e}", exc_info=False)
        return is_breakout, 0, ["Check error"], additional_info

    # Default reason if somehow empty
    if not reasons: reasons.append("No conditions met or error.")

    return is_breakout, confidence_score, reasons, additional_info

def analyze_stock(stock_ticker_ns):
    """ Fetches data, calculates indicators, performs breakout check. """
    if not stock_ticker_ns.endswith(('.NS', '.BO')):
        logging.warning(f"    Invalid ticker format: {stock_ticker_ns}")
        return None

    logging.info(f"  Analyzing {stock_ticker_ns}...")
    results = None # Default to None
    try:
        stock = yf.Ticker(stock_ticker_ns)
        # Fetch data: Use repair=True; consider prepost=False if not needed
        df = stock.history(period=HISTORICAL_PERIOD, interval="1d", auto_adjust=False, repair=True, prepost=False)

        if df.empty:
            logging.warning(f"    No {HISTORICAL_PERIOD} historical data for {stock_ticker_ns}.")
            return None
        if 'Volume' not in df.columns or df['Volume'].isnull().all() or (df['Volume'] == 0).all():
             logging.warning(f"    Insufficient/Invalid volume data for {stock_ticker_ns}.")
             return None

        # --- Data Cleaning ---
        df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
        # Forward fill Close for potential missing days before dropping NaNs in Volume
        df['Close'] = df['Close'].ffill()
        df.dropna(subset=['Close', 'Volume'], inplace=True)
        if df.empty:
             logging.warning(f"    Data invalid after cleaning for {stock_ticker_ns}.")
             return None

        # --- Calculate Indicators ---
        if not hasattr(df, 'ta'):
             logging.error(f"    Pandas TA not available for DataFrame on {stock_ticker_ns}.")
             return None

        df.ta.rsi(append=True)
        df.ta.macd(append=True)
        df.ta.sma(length=20, append=True)
        df.ta.sma(length=50, append=True)
        df.ta.bbands(length=20, std=2, append=True)
        vol_sma_col_name = f'SMA_{VOLUME_AVG_PERIOD}_VOL'
        # Ensure the column name is passed correctly for volume SMA
        df.ta.sma(close='Volume', length=VOLUME_AVG_PERIOD, append=True, col_names=(vol_sma_col_name,))

        # --- Perform Breakout Check ---
        is_breakout, confidence, reasons, additional_info = check_breakout_conditions(df.copy()) # Pass a copy to avoid modification issues

        # --- Calculate Overall Score ---
        score = float(confidence) # Base score on confidence
        last_row = df.iloc[-1].copy()
        rsi_value = last_row.get('RSI_14')
        if pd.notna(rsi_value):
            score += min(10, max(0, (rsi_value - 50) / 2)) # Add minor RSI boost

        # --- Gather Results ---
        try: # Added try block specifically around results creation for safety
            # --- Gather Results ---
            latest_data = df.iloc[-1].copy() # Use copy
            results = {
                'ticker': stock_ticker_ns, # String - OK
                'last_close': safe_json_convert(latest_data.get('Close')),
                'rsi': safe_json_convert(rsi_value), # rsi_value already extracted
                'macd': safe_json_convert(latest_data.get('MACD_12_26_9')),
                'macdsignal': safe_json_convert(latest_data.get('MACDs_12_26_9')),
                'macdhist': safe_json_convert(latest_data.get('MACDh_12_26_9')),
                'sma20': safe_json_convert(latest_data.get('SMA_20')),
                'sma50': safe_json_convert(latest_data.get('SMA_50')),
                'volume': safe_json_convert(latest_data.get('Volume')), # Convert Volume
                'avg_volume': safe_json_convert(latest_data.get(vol_sma_col_name)), # Convert Avg Vol
                'upper_bb': safe_json_convert(latest_data.get('BBU_20_2.0')),
                'breakout_signal': bool(is_breakout), # Ensure bool type
                'breakout_confidence': int(confidence), # Ensure int type
                'breakout_reasons': reasons,     # List of strings - OK
                'score': safe_json_convert(round(score, 1)) # Convert Score
            }
        except Exception as e:
            logging.error(f"Error while creating results for {stock_ticker_ns}: {e}", exc_info=False)
            results = None
        # logging.info(f"    Analysis complete for {stock_ticker_ns}: Score={score:.1f}, Brk={is_breakout}, Conf={confidence}%") # Verbose

    except Exception as e:
        logging.error(f"    ERROR Analyzing {stock_ticker_ns}: {e}", exc_info=False)
        # logging.debug(traceback.format_exc()) # Log trace if debugging needed

    return results # Return None if error occurred, otherwise the results dict

def run_pre_market_analysis():
    """ Orchestrates pre-market analysis, updates global state. """
    global monitored_stocks_tickers, pre_market_analysis_done_today
    global latest_sector_data, latest_top_stocks_data, last_analysis_time
    global latest_breakout_stocks_data  # Add this line to declare global variable
    current_analysis_start_time = datetime.now(IST)
    logging.info(f"--- Running Pre-Market Analysis ({current_analysis_start_time.strftime('%Y-%m-%d %H:%M:%S %Z')}) ---")

    # Use temporary variables to hold results during processing
    temp_monitored_tickers = []
    temp_sector_data = {}
    temp_top_stocks = []
    temp_breakout_stocks = []  # Temporary storage for breakout stocks

    try:
        # 1. Find top sector
        top_sectors = get_sector_performance()
        if not top_sectors:
            logging.warning("Could not determine top performing sectors.")
            raise ValueError("Sector performance calculation failed.")

        temp_sector_data = top_sectors
        top_sector_name = list(top_sectors.keys())[0]
        logging.info(f"Top Performing Sector: {top_sector_name} (Avg Perf: {top_sectors[top_sector_name].get('average_performance', 'N/A'):.2f}%)")

        # 2. Get stock list for the winning sector
        if top_sector_name not in indian_stocks_by_sector_py:
            logging.error(f"Sector '{top_sector_name}' not defined in map.")
            raise ValueError("Top sector not found in map.")
        sector_stock_dicts = indian_stocks_by_sector_py[top_sector_name]

        # 3. Prepare list of tickers to analyze
        tickers_to_analyze = [get_yahoo_ticker(stock['symbol']) for stock in sector_stock_dicts]
        logging.info(f"Analyzing {len(tickers_to_analyze)} stocks in '{top_sector_name}' sector...")

        # 4. Analyze stocks
        analysis_results = []
        processed_stock_count = 0
        for stock_ticker in tickers_to_analyze:
            processed_stock_count += 1
            result = analyze_stock(stock_ticker)
            if result:
                analysis_results.append(result)
            time.sleep(0.1)

        if not analysis_results:
             logging.warning(f"No stocks in {top_sector_name} could be analyzed successfully.")

        # 5. Rank stocks based on the overall analysis score
        ranked_stocks = sorted(analysis_results, key=lambda x: x.get('score', 0.0), reverse=True)
        temp_top_stocks = ranked_stocks[:TOP_N_STOCKS]

        # 6. Filter for Strong Breakouts
        STRONG_BREAKOUT_CONFIDENCE_THRESHOLD = 75  # Define how strong a breakout must be
        temp_breakout_stocks = [
            stock for stock in ranked_stocks
            if stock.get('breakout_signal') and stock.get('breakout_confidence', 0) >= STRONG_BREAKOUT_CONFIDENCE_THRESHOLD
        ]

        logging.info(f"Identified {len(temp_breakout_stocks)} strong breakout stocks.")

        # 7. Select Tickers for Monitoring Function (use Top Stocks, not Breakout necessarily)
        temp_monitored_tickers = [stock['ticker'] for stock in temp_top_stocks] # From top N Stocks not breakout

        logging.info(f"--- Pre-Market Analysis Complete (Duration: {(datetime.now(IST) - current_analysis_start_time).total_seconds():.1f}s) ---")
        logging.info(f"Selected Top {len(temp_monitored_tickers)} Stocks to Monitor (from {top_sector_name}):")
        # Log the selected stocks and their key metrics
        for i, stock_details in enumerate(temp_top_stocks): # Keep Top N
             ticker = stock_details['ticker']
             rsi_str = f"{stock_details.get('rsi', 'N/A'):.1f}" if pd.notna(stock_details.get('rsi')) else "N/A"
             confidence = stock_details.get('breakout_confidence', 0)
             is_breakout = stock_details.get('breakout_signal', False)
             reasons = stock_details.get('breakout_reasons', [])
             logging.info(f"  {i+1}. {ticker:<15} (Score: {stock_details.get('score', 0.0):<5.1f}, RSI: {rsi_str:<5}, Brk: {'Yes' if is_breakout else 'No'}{f' [{confidence}%]' if is_breakout else ''})")
             if is_breakout: logging.info(f"       Reasons: {'; '.join(reasons)}") # Keep same log
    except Exception as e:
        logging.error(f"Error during pre-market analysis execution: {e}", exc_info=True)
        temp_sector_data = latest_sector_data # Keep old sector data
        temp_top_stocks = []
        temp_monitored_tickers = []
        temp_breakout_stocks = []

    finally:
        # CRITICAL: Update Global State Safely - This block *always* runs
        print("Testing", temp_monitored_tickers)
        with api_lock:
            latest_sector_data = temp_sector_data
            latest_top_stocks_data = temp_top_stocks # Top N as before
            monitored_stocks_tickers = temp_monitored_tickers # Top N stocks from chosen sector.
            pre_market_analysis_done_today = True
            last_analysis_time = datetime.now(IST)
            logging.info("Global state updated after pre-market analysis attempt.")
            
def monitor_selected_stocks():
    """Fetches and logs the latest market data for the selected stocks."""
    current_monitoring_list = []
    with api_lock:  # Get a consistent list to monitor for this run
        if not monitored_stocks_tickers: return  # Exit if list is empty
        current_monitoring_list = list(monitored_stocks_tickers)

    if not current_monitoring_list: return  # Double check after lock release

    logging.info(f"--- Monitoring Update ({len(current_monitoring_list)} stocks) ---")
    for ticker_sym in current_monitoring_list:
        try:
            stock = yf.Ticker(ticker_sym)
            info = stock.fast_info # Use fast_info

            current_price = info.last_price
            prev_close = info.previous_close

            if pd.notna(current_price) and pd.notna(prev_close) and prev_close != 0:
                change = ((current_price - prev_close) / prev_close) * 100
                logging.info(f"  {ticker_sym:<15}: ?{current_price:<8.2f} ({change:+.2f}%)")
            elif pd.notna(current_price):
                logging.info(f"  {ticker_sym:<15}: ?{current_price:<8.2f} (Change N/A)")
            else:
                logging.warning(f"  {ticker_sym:<15}: Price not found.")
        except Exception as e:
            logging.warning(f"  Warn: Error fetching monitoring update for {ticker_sym}: {str(e)[:100]}", exc_info=False)
        time.sleep(0.1)  # Small delay between monitoring calls


# --- Scheduling Logic (Runs in its own thread) ---
def schedule_loop():
    """ Runs the scheduled tasks continuously, including periodic sector updates. """
    # Declare globals that might be modified within this loop's scope or called functions
    global pre_market_analysis_done_today, monitored_stocks_tickers, api_lock

    logging.info("Scheduler thread started.")

    def conditional_pre_market_analysis():
        """Wrapper to check time of day before running pre-market analysis."""
        now_ist = datetime.now(IST)
        if datetime.strptime("08:45", "%H:%M").time() <= now_ist.time() < datetime.strptime("15:30", "%H:%M").time():
            run_pre_market_analysis()

    schedule.every(30).minutes.do(conditional_pre_market_analysis)

    # --- Schedule News Fetching ---
    # Runs periodically to fetch the latest news headlines
    logging.info(f"Scheduling news fetching task every {NEWS_FETCH_INTERVAL_MINUTES} minutes.")
    schedule.every(NEWS_FETCH_INTERVAL_MINUTES).minutes.do(fetch_and_store_latest_news)

    # --- Schedule Stock Monitoring (Conditional) ---
    # Runs every 15 minutes, but only executes logic if market is open
    logging.info(f"Scheduling stock monitoring task every 15 minutes (conditionally during market hours).")
    def conditional_monitor():
        """Wrapper to check market hours before monitoring."""
        now_ist = datetime.now(IST)
        # Ensure MARKET_OPEN_TIME and MARKET_CLOSE_TIME are defined correctly globally
        if MARKET_OPEN_TIME <= now_ist.time() < MARKET_CLOSE_TIME:
             logging.debug("Market open, running monitor_selected_stocks.") # Debug level log
             monitor_selected_stocks()
        # else:
        #     logging.debug("Market closed, skipping monitor_selected_stocks.") # Optional debug log
    schedule.every(15).minutes.do(conditional_monitor)


    # --- Run Initial Tasks on Startup (in background threads) ---
    # This prevents blocking the start of the Flask server if analysis takes time
    logging.info("Triggering initial tasks on startup (in background)...")

    # Initial Pre-Market Analysis (important for initial API state)
    initial_analysis_thread = threading.Thread(
        target=run_pre_market_analysis,
        name="InitialAnalysisThread",
        daemon=True
    )
    initial_analysis_thread.start()

    # Initial News Fetch
    initial_news_thread = threading.Thread(
        target=fetch_and_store_latest_news,
        name="InitialNewsThread",
        daemon=True
    )
    initial_news_thread.start()

    # Optionally run initial sector update too? Uncomment if needed.
    # logging.info("Triggering initial sector performance update...")
    # initial_sector_update_thread = threading.Thread(
    #     target=run_periodic_sector_update,
    #     name="InitialSectorUpdateThread",
    #     daemon=True
    # )
    # initial_sector_update_thread.start()

    logging.info("Initial tasks launched.")
    # --- End Initial Tasks ---


    # --- Main Scheduler Loop ---
    while True:
        try:
            # Execute any pending scheduled jobs
            schedule.run_pending()

            # --- Reset flags logic after market close ---
            now_ist = datetime.now(IST)
            current_time_ist = now_ist.time()
            # Use a time slightly after market close for reset
            reset_time = datetime.strptime("16:05", "%H:%M").time()

            # Only perform reset actions if the flag indicates analysis was done *and* it's past reset time
            if current_time_ist >= reset_time and pre_market_analysis_done_today:
                # Use lock to safely modify shared global variables
                with api_lock:
                    # Check again inside lock to prevent race conditions if needed,
                    # though pre_market_analysis_done_today check outside is likely sufficient here.
                    if monitored_stocks_tickers: # Only log reset if we were monitoring
                        logging.info(f"Market closed. Resetting pre-market flag and clearing monitored stocks.")
                        monitored_stocks_tickers = [] # Clear the list for the next day

                    # Always reset the flag after the first check past reset time
                    pre_market_analysis_done_today = False
            # --- End Reset Logic ---

        except Exception as e:
            # Log errors occurring within the schedule loop itself or during job execution
            logging.error("Error within schedule_loop execution:", exc_info=True)
            # Consider adding a small delay here if errors are repeating rapidly
            time.sleep(5) # Wait 5 seconds after an error in the loop

        # Sleep before checking the schedule again
        # A longer sleep is fine as schedule handles the exact timing
        time.sleep(20) # Check schedule every 20 seconds is generally sufficient

# --- Main Execution ---
if __name__ == "__main__":
    required_modules = ['pytz', 'schedule', 'pandas_ta', 'statistics', 'numpy', 'flask', 'flask_cors', 'threading', 'yfinance', 'pandas', 'feedparser']
    try:
        modules = {}
        for module_name in required_modules:
            modules[module_name] = __import__(module_name)
        logging.info(f"Using pandas_ta version: {modules['pandas_ta'].version}")
        logging.info(f"Using yfinance version: {modules['yfinance'].__version__}")
        logging.info(f"Using Flask version: {modules['flask'].__version__}")
    except ImportError as e:
        logging.critical(f"CRITICAL ERROR: Missing required library - {e.name}")
        print(f"Install required libraries: pip install {' '.join(required_modules)}")
        exit(1) # Exit with error code

    logging.info("="*50)
    logging.info(" Starting Indian Stock Monitor Backend ")
    logging.info("="*50)

    flask_server_port = 8080 # Standard port, can be changed

    try:
        # Start the scheduler loop in a separate thread
        # Mark as daemon so it exits when the main thread (Flask) exits
        # Run pre-market analysis on initial start
        scheduler_thread = threading.Thread(target=schedule_loop, name="SchedulerThread", daemon=True)
        scheduler_thread.start()
        logging.info("Scheduler thread launched.")

        # Start the Flask API server in the main thread
        logging.info(f"Starting Flask API server on http://0.0.0.0:{flask_server_port} ...")
        logging.info("Use Ctrl+C to stop the server.")
        # Use use_reloader=False for stability with threads/scheduling
        app.run(host='0.0.0.0', port=flask_server_port, debug=False, use_reloader=False)

    except KeyboardInterrupt:
        logging.info("\nCtrl+C received. Shutting down gracefully.")
    except Exception as e:
        logging.critical(f"FATAL ERROR preventing server start or during runtime", exc_info=True)
        print(f"\nFATAL ERROR: {e}")
        exit(1)

    logging.info("Backend server stopped.")