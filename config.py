# --- IMPORTS ---
import pytz

# --- Setup Logging ---
# Configure logging to show timestamp, level, thread name, and message
LOGGING_LEVEL = 'INFO'
LOGGING_FORMAT = '%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s'
LOGGING_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

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
MARKET_OPEN_TIME = pytz.time(9, 15, 0) # 09:15
MARKET_CLOSE_TIME = pytz.time(15, 30, 0) # 15:30
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
FLASK_SERVER_PORT = 8080 # Standard port, can be changed