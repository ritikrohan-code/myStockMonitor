import yfinance as yf
import pandas as pd
import pandas_ta as ta
import schedule
import time
from datetime import datetime
import pytz # For timezone handling
import statistics # For calculating mean easily
import numpy as np # For checking NaN

# --- Configuration based on your map ---
# [Keep the indian_stocks_by_sector_py dictionary as is]
indian_stocks_by_sector_py = {
  "IT": [
    { "symbol": "TCS", "name": "Tata Consultancy Services" },
    { "symbol": "INFY", "name": "Infosys" },
    { "symbol": "WIPRO", "name": "Wipro" },
    { "symbol": "HCLTECH", "name": "HCL Technologies" },
    { "symbol": "TECHM", "name": "Tech Mahindra" },
    { "symbol": "LTIM", "name": "LTI Mindtree" },
    { "symbol": "MPHASIS", "name": "Mphasis" },
    { "symbol": "COFORGE", "name": "Coforge" }
  ],
  "Banking": [
    { "symbol": "HDFCBANK", "name": "HDFC Bank" },
    { "symbol": "ICICIBANK", "name": "ICICI Bank" },
    { "symbol": "SBIN", "name": "State Bank of India" },
    { "symbol": "KOTAKBANK", "name": "Kotak Mahindra Bank" },
    { "symbol": "AXISBANK", "name": "Axis Bank" },
    { "symbol": "INDUSINDBK", "name": "IndusInd Bank" },
    { "symbol": "BANDHANBNK", "name": "Bandhan Bank" },
    { "symbol": "FEDERALBNK", "name": "Federal Bank" }
  ],
  "Financial Services": [
    { "symbol": "BAJFINANCE", "name": "Bajaj Finance" },
    { "symbol": "SBILIFE", "name": "SBI Life Insurance" },
    { "symbol": "HDFCLIFE", "name": "HDFC Life Insurance" },
    { "symbol": "ICICIPRULI", "name": "ICICI Prudential Life Insurance" },
    { "symbol": "BAJAJFINSV", "name": "Bajaj Finserv" },
    { "symbol": "RECLTD", "name": "REC Limited" },
    { "symbol": "PFC", "name": "Power Finance Corporation" }
  ],
  "Automobile": [
    { "symbol": "MARUTI", "name": "Maruti Suzuki" },
    { "symbol": "TATAMOTORS", "name": "Tata Motors" },
    { "symbol": "M&M", "name": "Mahindra & Mahindra" },
    { "symbol": "BAJAJ-AUTO", "name": "Bajaj Auto" },
    { "symbol": "EICHERMOT", "name": "Eicher Motors" },
    { "symbol": "HEROMOTOCO", "name": "Hero MotoCorp" },
    { "symbol": "TVSMOTOR", "name": "TVS Motor Company" },
    { "symbol": "ASHOKLEY", "name": "Ashok Leyland" }
  ],
  "Pharmaceuticals": [
    { "symbol": "SUNPHARMA", "name": "Sun Pharmaceutical" },
    { "symbol": "DRREDDY", "name": "Dr. Reddy's Laboratories" },
    { "symbol": "CIPLA", "name": "Cipla" },
    { "symbol": "DIVISLAB", "name": "Divi's Laboratories" },
    { "symbol": "LUPIN", "name": "Lupin" },
    { "symbol": "AUROPHARMA", "name": "Aurobindo Pharma" },
    { "symbol": "BIOCON", "name": "Biocon" },
    { "symbol": "TORNTPHARM", "name": "Torrent Pharmaceuticals" }
  ],
  "Energy": [
    { "symbol": "RELIANCE", "name": "Reliance Industries" },
    { "symbol": "ONGC", "name": "Oil & Natural Gas Corporation" },
    { "symbol": "IOC", "name": "Indian Oil Corporation" },
    { "symbol": "BPCL", "name": "Bharat Petroleum" },
    { "symbol": "HINDPETRO", "name": "Hindustan Petroleum" },
    { "symbol": "GAIL", "name": "GAIL India" },
    { "symbol": "NTPC", "name": "NTPC" },
    { "symbol": "POWERGRID", "name": "Power Grid Corporation" }
  ],
  "Metals & Mining": [
    { "symbol": "TATASTEEL", "name": "Tata Steel" },
    { "symbol": "JSWSTEEL", "name": "JSW Steel" },
    { "symbol": "HINDALCO", "name": "Hindalco Industries" },
    { "symbol": "VEDL", "name": "Vedanta Limited" },
    { "symbol": "COALINDIA", "name": "Coal India" },
    { "symbol": "SAIL", "name": "Steel Authority of India" },
    { "symbol": "NMDC", "name": "NMDC" },
    { "symbol": "HINDZINC", "name": "Hindustan Zinc" }
  ],
  "FMCG": [
    { "symbol": "ITC", "name": "ITC Limited" },
    { "symbol": "HINDUNILVR", "name": "Hindustan Unilever" },
    { "symbol": "NESTLEIND", "name": "Nestle India" },
    { "symbol": "BRITANNIA", "name": "Britannia Industries" },
    { "symbol": "DABUR", "name": "Dabur India" },
    { "symbol": "GODREJCP", "name": "Godrej Consumer Products" },
    { "symbol": "MARICO", "name": "Marico" },
    { "symbol": "COLPAL", "name": "Colgate-Palmolive" }
  ],
  "Telecom": [
    { "symbol": "BHARTIARTL", "name": "Bharti Airtel" },
    { "symbol": "VODAFONEIDEA", "name": "Vodafone Idea" }, # Use IDEA.NS
    { "symbol": "MTNL", "name": "MTNL" }
  ],
  "Infrastructure": [
    { "symbol": "LT", "name": "Larsen & Toubro" },
    { "symbol": "ADANIPORTS", "name": "Adani Ports" },
    { "symbol": "IRCTC", "name": "Indian Railway Catering" },
    { "symbol": "CONCOR", "name": "Container Corporation" },
    { "symbol": "NBCC", "name": "NBCC" },
    { "symbol": "BHEL", "name": "Bharat Heavy Electricals" },
    { "symbol": "HUDCO", "name": "Housing & Urban Development" }
  ],
  "Consumer Durables": [
    { "symbol": "TITAN", "name": "Titan Company" },
    { "symbol": "VOLTAS", "name": "Voltas" },
    { "symbol": "BLUESTARCO", "name": "Blue Star" },
    { "symbol": "CROMPTON", "name": "Crompton Greaves" },
    { "symbol": "HAVELLS", "name": "Havells India" },
    { "symbol": "WHIRLPOOL", "name": "Whirlpool India" }
  ],
  "Chemicals": [
    { "symbol": "UPL", "name": "UPL Limited" },
    { "symbol": "SRF", "name": "SRF Limited" },
    { "symbol": "PIIND", "name": "PI Industries" },
    { "symbol": "TATACHEM", "name": "Tata Chemicals" },
    { "symbol": "GNFC", "name": "Gujarat Narmada Valley" },
    { "symbol": "FINEORG", "name": "Fine Organic" }
  ],
  "Real Estate": [
    { "symbol": "DLF", "name": "DLF Limited" },
    { "symbol": "GODREJPROP", "name": "Godrej Properties" },
    { "symbol": "PRESTIGE", "name": "Prestige Estates" },
    { "symbol": "SOBHA", "name": "Sobha Limited" },
    { "symbol": "BRIGADE", "name": "Brigade Enterprises" }
  ],
  "Media & Entertainment": [
    { "symbol": "ZEEL", "name": "Zee Entertainment" },
    { "symbol": "SUNTV", "name": "Sun TV Network" },
    { "symbol": "PVRINOX", "name": "PVR INOX" }
  ],
  "Gold & Jewellery": [
    { "symbol": "TITAN", "name": "Titan Company" },
    { "symbol": "RAJESHEXPO", "name": "Rajesh Exports" },
    { "symbol": "PCJEWELLER", "name": "PC Jeweller" },
    { "symbol": "TBZ", "name": "Tribhovandas Bhimji Zaveri" },
    { "symbol": "GOLDIAM", "name": "Goldiam International" },
    { "symbol": "THANGAMAYL", "name": "Thangamayil Jewellery" },
    { "symbol": "SENCO", "name": "Senco Gold" },
    { "symbol": "MUTHOOTFIN", "name": "Muthoot Finance" },
    { "symbol": "MANAPPURAM", "name": "Manappuram Finance" },
    { "symbol": "IIFL", "name": "IIFL Finance" }
 ]
}

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

# Global variables
monitored_stocks = []
pre_market_analysis_done_today = False

# --- Functions ---

def get_sector_performance():
    """
    Calculates average performance of sectors based on the constituent stocks
    provided in the indian_stocks_by_sector_py map.
    Uses previous day's close for calculation.
    Returns a dictionary sorted by average performance (highest first).
    """
    print("Calculating sector performance by averaging constituent stocks...")
    sector_avg_performance = {}

    for sector_name, stock_list in indian_stocks_by_sector_py.items():
        print(f"  Processing Sector: {sector_name} ({len(stock_list)} stocks)")
        sector_stock_performances = []

        for stock_info in stock_list:
            symbol = stock_info['symbol']
            # --- Handle Yahoo Finance Ticker Formatting ---
            if symbol == "M&M": ticker_ns = "M&M.NS"
            elif symbol == "VODAFONEIDEA": ticker_ns = "IDEA.NS"
            elif symbol == "IIFL": ticker_ns = "IIFL.NS"
            elif '-' in symbol: ticker_ns = symbol + ".NS"
            else: ticker_ns = symbol + ".NS"
            # --- End Ticker Formatting ---

            # print(f"    - Fetching {ticker_ns}...", end="") # Verbose, can be commented out
            try:
                stock = yf.Ticker(ticker_ns)
                hist = stock.history(period="3d", auto_adjust=False)

                if len(hist) >= 2:
                    last_close = hist['Close'].iloc[-1]
                    prev_close = hist['Close'].iloc[-2]
                    if pd.notna(last_close) and pd.notna(prev_close) and prev_close != 0:
                        performance = ((last_close - prev_close) / prev_close) * 100
                        sector_stock_performances.append(performance)
                        # print(f" Perf: {performance:.2f}%") # Verbose
                    # else: print(" Invalid data. Skipping.") # Verbose
                # else: print(" Insufficient history. Skipping.") # Verbose
            except Exception as e:
                 # Suppress individual stock errors in this summary function
                 pass # print(f" ERROR fetching {ticker_ns}. Skipping. ({e})")
            time.sleep(0.2) # Slightly reduce sleep if less printing

        if sector_stock_performances:
            average_perf = statistics.mean(sector_stock_performances)
            sector_avg_performance[sector_name] = {
                'average_performance': average_perf,
                'stocks_calculated': len(sector_stock_performances),
                'stocks_defined': len(stock_list)
            }
            print(f"  Sector {sector_name} Avg Performance: {average_perf:.2f}% ({len(sector_stock_performances)}/{len(stock_list)} stocks)")
        else:
            print(f"  Sector {sector_name}: Could not calculate performance.")

    print("\nSorting sectors by calculated average performance...")
    sorted_sectors = dict(sorted(
        sector_avg_performance.items(),
        key=lambda item: item[1]['average_performance'],
        reverse=True
    ))
    return sorted_sectors

# --- REVAMPED Breakout Analysis Function ---
def check_breakout_conditions(df):
    """
    Analyzes DataFrame for complex breakout conditions including Volume Surge and BBands.

    Args:
        df (pd.DataFrame): DataFrame with OHLCV and calculated indicators:
                           SMA_50, RSI_14, MACD_12_26_9, MACDs_12_26_9,
                           BBU_20_2.0 (Upper BBand), SMA_VOL_20 (Avg Volume).

    Returns:
        tuple: (is_breakout (bool), confidence_score (int), reasons (list))
    """
    is_breakout = False
    achieved_score = 0
    confidence_score = 0
    reasons = []

    # --- Constants for Scoring ---
    POINTS_PRICE_VS_SMA50 = 15
    POINTS_PRICE_VS_BBAND = 25 # Higher weight for BB breakout
    POINTS_VOLUME_SURGE = 30   # Highest weight for volume confirmation
    POINTS_RSI = 15
    POINTS_MACD_CROSS = 15
    # Total points should match MAX_POSSIBLE_BREAKOUT_POINTS if all are met = 100

    # Ensure DataFrame has enough data and required columns
    min_rows_needed = max(51, VOLUME_AVG_PERIOD + 1) # Need enough for SMAs and Vol Avg
    if df is None or df.empty or len(df) < min_rows_needed:
        return is_breakout, confidence_score, ["Insufficient data"]

    required_cols = ['Close', 'Volume', 'SMA_50', 'RSI_14',
                     'MACD_12_26_9', 'MACDs_12_26_9',
                     'BBU_20_2.0', f'SMA_{VOLUME_AVG_PERIOD}_VOL'] # Ensure correct volume SMA col name
    if not all(col in df.columns and pd.notna(df[col].iloc[-1]) for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns or pd.isna(df[col].iloc[-1])]
        # Don't print warning for every stock, return quietly
        return is_breakout, confidence_score, [f"Missing indicators: {missing}"]

    try:
        # Get latest values
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

        # 1. Price vs SMA50
        if last_close > sma50:
            achieved_score += POINTS_PRICE_VS_SMA50
            reasons.append(f"Price > SMA50 ({last_close:.2f} > {sma50:.2f})")

        # 2. Price vs Upper Bollinger Band
        if last_close > upper_bb:
            achieved_score += POINTS_PRICE_VS_BBAND
            reasons.append(f"Price > Upper BB ({last_close:.2f} > {upper_bb:.2f})")

        # 3. Volume Surge
        if avg_vol > 0 and last_vol > (avg_vol * VOLUME_SURGE_MULTIPLIER):
             achieved_score += POINTS_VOLUME_SURGE
             vol_ratio = last_vol / avg_vol
             reasons.append(f"Volume Surge ({last_vol:,.0f} vs Avg {avg_vol:,.0f} = {vol_ratio:.1f}x)")

        # 4. RSI Strength
        if rsi > RSI_BREAKOUT_THRESHOLD:
            achieved_score += POINTS_RSI
            reasons.append(f"RSI > {RSI_BREAKOUT_THRESHOLD} ({rsi:.1f})")

        # 5. MACD Bullish Cross
        if macd_line > macd_signal:
            achieved_score += POINTS_MACD_CROSS
            reasons.append(f"MACD Line > Signal ({macd_line:.2f} > {macd_signal:.2f})")

        # --- Determine Final Breakout Status and Confidence ---
        if achieved_score >= MINIMUM_BREAKOUT_SCORE_THRESHOLD:
            is_breakout = True
            # Normalize score to 0-100 range
            confidence_score = min(100, int(round((achieved_score / MAX_POSSIBLE_BREAKOUT_POINTS) * 100)))
        else:
             # If below threshold, no breakout, confidence is 0
             is_breakout = False
             confidence_score = 0
             reasons = ["Score below threshold"] # Clear previous reasons if no breakout

    except KeyError as e:
        return is_breakout, confidence_score, [f"Missing column '{e}'"]
    except Exception as e:
        return is_breakout, confidence_score, [f"Error in breakout check: {e}"]

    # Ensure reasons list is not empty if breakout is True
    if is_breakout and not reasons:
       reasons.append("Conditions met, but reason generation failed.")
    elif not is_breakout and not reasons:
       reasons.append("No breakout conditions met.")


    return is_breakout, confidence_score, reasons
# --- END Breakout Analysis Function ---


def analyze_stock(stock_ticker_ns):
    """
    Fetches 1y data, calculates indicators including BBands & Vol SMA,
    performs complex breakout check, returns detailed analysis.
    """
    if not stock_ticker_ns.endswith(('.NS', '.BO')):
        print(f"    Skipping analysis for '{stock_ticker_ns}' - invalid ticker format.")
        return None

    try:
        print(f"  Analyzing {stock_ticker_ns} (1y data)...")
        stock = yf.Ticker(stock_ticker_ns)
        # Fetch data first
        df = stock.history(period=HISTORICAL_PERIOD, interval="1d", auto_adjust=False) # Use auto_adjust=False for Volume

        if df.empty:
            print(f"    No {HISTORICAL_PERIOD} historical data found for {stock_ticker_ns}.")
            return None
        if 'Volume' not in df.columns or df['Volume'].sum() == 0:
            print(f"    No volume data found for {stock_ticker_ns}. Skipping analysis.")
            return None # Cannot calculate volume SMA or check surge

        # Ensure Volume is numeric, coercing errors
        df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
        df.dropna(subset=['Volume'], inplace=True) # Drop rows where volume became NaN
        if df.empty:
             print(f"    Volume data invalid after cleaning for {stock_ticker_ns}. Skipping.")
             return None


        if not hasattr(df, 'ta'):
             print(f"    ERROR: DataFrame for {stock_ticker_ns} missing '.ta' attribute.")
             return None

        # Calculate Technical Indicators (including new ones)
        print(f"    Calculating indicators for {stock_ticker_ns}...")
        df.ta.rsi(append=True)
        df.ta.macd(append=True)
        df.ta.sma(length=20, append=True)
        df.ta.sma(length=50, append=True)
        df.ta.bbands(length=20, std=2, append=True) # Calculate Bollinger Bands
        # Calculate Volume SMA - IMPORTANT: Use 'Volume' column
        vol_sma_col_name = f'SMA_{VOLUME_AVG_PERIOD}_VOL'
        df.ta.sma(close='Volume', length=VOLUME_AVG_PERIOD, append=True, col_names=(vol_sma_col_name,))

        # --- Perform Complex Breakout Check ---
        print(f"    Checking breakout conditions for {stock_ticker_ns}...")
        is_breakout, confidence, reasons = check_breakout_conditions(df)

        # --- Calculate Overall Score ---
        # Make score heavily influenced by breakout confidence
        score = float(confidence) # Start score with confidence
        last_row = df.iloc[-1]
        rsi_value = last_row.get('RSI_14')

        # Add minor adjustment based on RSI (e.g., +0 to +10 points)
        if pd.notna(rsi_value):
            score += min(10, max(0, (rsi_value - 50) / 2)) # Scale RSI contribution

        # --- Gather Results ---
        latest_data = df.iloc[-1]
        results = {
            'ticker': stock_ticker_ns,
            'last_close': latest_data.get('Close'),
            'rsi': rsi_value,
            'macd': latest_data.get('MACD_12_26_9'),
            'macdsignal': latest_data.get('MACDs_12_26_9'),
            'macdhist': latest_data.get('MACDh_12_26_9'),
            'sma20': latest_data.get('SMA_20'),
            'sma50': latest_data.get('SMA_50'),
            'volume': latest_data.get('Volume'),
            'avg_volume': latest_data.get(vol_sma_col_name),
            'upper_bb': latest_data.get('BBU_20_2.0'),
            'breakout_signal': is_breakout,
            'breakout_confidence': confidence, # Store confidence
            'breakout_reasons': reasons,     # Store reasons
            'score': score # Overall score, driven by confidence
        }
        print(f"    Analysis complete for {stock_ticker_ns}: Score={score:.1f}, Breakout={is_breakout}, Confidence={confidence}%")
        return results

    except Exception as e:
        print(f"    ERROR Analyzing {stock_ticker_ns}: {e}")
        import traceback
        traceback.print_exc() # Print detailed trace for debugging analysis errors
        return None


def run_pre_market_analysis():
    """ Orchestrates pre-market analysis, finds top sector, analyzes stocks, selects top N based on score. """
    global monitored_stocks, pre_market_analysis_done_today
    print(f"\n--- Running Pre-Market Analysis ({datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S %Z')}) ---")

    # 1. Find top performing sector
    top_sectors = get_sector_performance()
    if not top_sectors:
        print("Could not determine top performing sectors. Aborting.")
        return

    top_sector_name = list(top_sectors.keys())[0]
    top_sector_info = top_sectors[top_sector_name]
    print(f"\nTop Performing Sector: {top_sector_name} (Avg Perf: {top_sector_info.get('average_performance', 'N/A'):.2f}%)")

    # 2. Get stock list for the winning sector
    if top_sector_name not in indian_stocks_by_sector_py:
        print(f"Error: Sector '{top_sector_name}' not defined in map.")
        return
    sector_stock_dicts = indian_stocks_by_sector_py[top_sector_name]

    # 3. Prepare list of tickers (.NS suffix) to analyze
    tickers_to_analyze = []
    for stock_info in sector_stock_dicts:
         symbol = stock_info['symbol']
         # --- Apply .NS suffix consistently ---
         if symbol == "M&M": ticker_ns = "M&M.NS"
         elif symbol == "VODAFONEIDEA": ticker_ns = "IDEA.NS"
         elif symbol == "IIFL": ticker_ns = "IIFL.NS" # Verify correct IIFL ticker
         elif '-' in symbol: ticker_ns = symbol + ".NS"
         else: ticker_ns = symbol + ".NS"
         tickers_to_analyze.append(ticker_ns)

    print(f"\nAnalyzing {len(tickers_to_analyze)} stocks defined in the '{top_sector_name}' sector...")

    # 4. Analyze these specific stocks
    analysis_results = []
    for stock_ticker in tickers_to_analyze:
        result = analyze_stock(stock_ticker)
        if result:
            analysis_results.append(result)
        time.sleep(0.5) # Shorter sleep needed if analysis takes longer

    # 5. Rank stocks based on the overall analysis score
    ranked_stocks = sorted(analysis_results, key=lambda x: x.get('score', 0.0), reverse=True)

    # 6. Select Top N stocks
    monitored_stocks = [stock['ticker'] for stock in ranked_stocks[:TOP_N_STOCKS]]

    print("\n--- Pre-Market Analysis Complete ---")
    print(f"Selected Top {TOP_N_STOCKS} Stocks to Monitor (from {top_sector_name}):")
    if monitored_stocks:
        for i, ticker in enumerate(monitored_stocks):
             stock_details = next((s for s in ranked_stocks if s['ticker'] == ticker), None)
             if stock_details:
                 rsi_str = f"{stock_details.get('rsi', 'N/A'):.1f}" if pd.notna(stock_details.get('rsi')) else "N/A"
                 confidence = stock_details.get('breakout_confidence', 0)
                 is_breakout = stock_details.get('breakout_signal', False)
                 reasons = stock_details.get('breakout_reasons', [])

                 # Basic Print Line
                 print(f"  {i+1}. {ticker:<15} "
                       f"(Score: {stock_details.get('score', 0.0):<5.1f}, "
                       f"RSI: {rsi_str:<5}, "
                       f"Breakout: {'Yes' if is_breakout else 'No'}{f' [{confidence}%]' if is_breakout else ''})")

                 # Print Reasons ONLY if it's a breakout
                 if is_breakout:
                     print(f"       Reasons: {'; '.join(reasons)}")

             else:
                 print(f"  {i+1}. {ticker}")
    else:
        print("  No stocks selected based on analysis criteria.")

    pre_market_analysis_done_today = True


def monitor_selected_stocks():
    """ Fetches and prints the latest 15m interval data for the selected stocks. """
    global monitored_stocks
    if not monitored_stocks:
        return

    print(f"\n--- Monitoring Update ({datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S %Z')}) ---")
    for ticker_sym in monitored_stocks:
        try:
            stock = yf.Ticker(ticker_sym)
            info = stock.info
            current_price = info.get('currentPrice') or info.get('regularMarketPrice')
            prev_close = info.get('previousClose') or info.get('regularMarketPreviousClose')

            if current_price and prev_close:
                 change = ((current_price - prev_close) / prev_close) * 100
                 print(f"  {ticker_sym:<15}: ₹{current_price:<8.2f} ({change:+.2f}%)")
            elif current_price:
                 print(f"  {ticker_sym:<15}: ₹{current_price:<8.2f} (Change N/A)")
            else:
                 print(f"  {ticker_sym:<15}: .info missing price, fetching {INTRADAY_INTERVAL} history...", end="")
                 hist = stock.history(period="1d", interval=INTRADAY_INTERVAL, auto_adjust=True)
                 if not hist.empty:
                     last_data = hist.iloc[-1]
                     last_close_hist = last_data['Close']
                     if prev_close and pd.notna(last_close_hist):
                         change_hist = ((last_close_hist - prev_close) / prev_close) * 100
                         change_str = f"({change_hist:+.2f}%)"
                     else:
                         change_str = "(Change N/A)"
                     timestamp_str = hist.index[-1].astimezone(IST).strftime('%H:%M')
                     print(f" ₹{last_close_hist:<8.2f} {change_str} (at {timestamp_str})")
                 else:
                     print(" Could not retrieve latest data.")
        except Exception as e:
            print(f"  Error fetching update for {ticker_sym}: {e}")
        time.sleep(0.3) # Shorter sleep for monitoring updates


# --- Scheduling Logic ---

def main_loop():
    """ Main operational loop checking time and running scheduled tasks for IST. """
    global pre_market_analysis_done_today, monitored_stocks

    print("Running initial pre-market analysis on startup...")
    run_pre_market_analysis()
    print("-" * 30)

    schedule_time = "08:45"
    print(f"Scheduling daily pre-market analysis at {schedule_time} IST.")
    schedule.every().day.at(schedule_time).do(run_pre_market_analysis)

    print(f"Scheduling monitoring task every 15 minutes during market hours (uses {INTRADAY_INTERVAL} data).")
    schedule.every(15).minutes.do(monitor_selected_stocks) # Changed schedule to 15 mins

    print("-" * 30)
    print("Scheduler started. Waiting for scheduled tasks...")
    print(f"Market Hours (IST): {MARKET_OPEN_TIME.strftime('%H:%M')} - {MARKET_CLOSE_TIME.strftime('%H:%M')}")
    print(f"Currently Monitoring: {monitored_stocks if monitored_stocks else 'None'}")
    print("-" * 30)

    while True:
        now_ist = datetime.now(IST)
        current_time_ist = now_ist.time()

        reset_time = datetime.strptime("16:00", "%H:%M").time()
        if current_time_ist >= reset_time and pre_market_analysis_done_today:
             if monitored_stocks:
                 print(f"\nMarket closed. Resetting pre-market flag for tomorrow ({now_ist.strftime('%Y-%m-%d %H:%M:%S %Z')}).")
                 monitored_stocks = []
             pre_market_analysis_done_today = False

        schedule.run_pending()
        time.sleep(30)

# --- Main Execution ---
if __name__ == "__main__":
    try:
        import pytz, schedule, pandas_ta, statistics, numpy
        print(f"Using pandas_ta version: {pandas_ta.version}")
    except ImportError as e:
        print(f"Error: Missing required library - {e.name}")
        print("Install required libraries: pip install yfinance pandas pandas-ta schedule pytz numpy")
        exit()

    print("="*50)
    print(" Starting Indian Stock Monitor with Custom Sector Map ")
    print("="*50)

    try:
        main_loop()
    except KeyboardInterrupt:
        print("\nExiting application.")
    except Exception as e:
        print(f"\nFATAL ERROR in main loop: {e}")
        import traceback
        traceback.print_exc()