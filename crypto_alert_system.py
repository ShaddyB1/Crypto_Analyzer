import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import schedule
import time
import requests
import logging
import logging.handlers
from typing import List, Dict, Tuple, Optional
from scipy import stats
import talib
import os
import signal
import sys
import traceback
from dataclasses import dataclass
import subprocess
import sqlite3
import json
import threading
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle

@dataclass
class PredictionRecord:
    """Store prediction data and outcomes"""
    timestamp: str
    coin_symbol: str
    prediction_type: str  # 'price_movement', 'risk_level', etc.
    predicted_value: float
    actual_value: float
    features: Dict[str, float]
    was_correct: bool

@dataclass
class BTCAnalysis:
    """Bitcoin analysis data container"""
    price: float
    price_history: List[float]
    change_24h: float
    change_7d: float
    dominance: float
    correlation: Dict[str, float]
    technical_indicators: Dict[str, float]
    market_sentiment: str
    support_levels: List[float]
    resistance_levels: List[float]
    prediction_confidence: float = 0.0

class PredictionTracker:
    """Manage and learn from prediction history"""
    def __init__(self, db_path: str = 'crypto_predictions.db'):
        self.db_path = db_path
        self.setup_database()
        self.model = None
        self.scaler = StandardScaler()
        
    def setup_database(self):
        """Initialize the database tables"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY,
                    timestamp TEXT,
                    coin_symbol TEXT,
                    prediction_type TEXT,
                    predicted_value REAL,
                    actual_value REAL,
                    features TEXT,
                    was_correct INTEGER
                )
            ''')
            
    def record_prediction(self, record: PredictionRecord):
        """Store a new prediction"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO predictions 
                (timestamp, coin_symbol, prediction_type, predicted_value, 
                actual_value, features, was_correct)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                record.timestamp,
                record.coin_symbol,
                record.prediction_type,
                record.predicted_value,
                record.actual_value,
                json.dumps(record.features),
                int(record.was_correct)
            ))

    def update_prediction_outcome(self, timestamp: str, coin_symbol: str, 
                                actual_value: float):
        """Update a prediction with its actual outcome"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT predicted_value, features
                FROM predictions
                WHERE timestamp = ? AND coin_symbol = ?
                ORDER BY id DESC LIMIT 1
            ''', (timestamp, coin_symbol))
            
            row = cursor.fetchone()
            if row:
                predicted_value, features = row
                was_correct = self._evaluate_prediction(
                    predicted_value, actual_value
                )
                
                conn.execute('''
                    UPDATE predictions
                    SET actual_value = ?, was_correct = ?
                    WHERE timestamp = ? AND coin_symbol = ?
                ''', (actual_value, int(was_correct), timestamp, coin_symbol))
                
    def _evaluate_prediction(self, predicted: float, actual: float) -> bool:
        """Determine if a prediction was correct"""
        # Consider a prediction correct if it's within 5% of actual value
        margin = abs(predicted * 0.05)
        return abs(predicted - actual) <= margin
        
    def train_model(self):
        """Train the prediction model on historical data"""
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql('''
                SELECT * FROM predictions 
                WHERE actual_value IS NOT NULL
            ''', conn)
            
        if len(df) < 100:  # Wait for sufficient data
            return
            
        # Prepare features
        X = pd.DataFrame([json.loads(f) for f in df['features']])
        y = df['was_correct'].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model = RandomForestClassifier(n_estimators=100)
        self.model.fit(X_scaled, y)
        
    def get_prediction_confidence(self, features: Dict[str, float]) -> float:
        """Get confidence score for a new prediction"""
        if self.model is None:
            return 0.5  # Default confidence when no model exists
            
        features_df = pd.DataFrame([features])
        features_scaled = self.scaler.transform(features_df)
        
        # Get probability of success
        proba = self.model.predict_proba(features_scaled)[0]
        return proba[1]  # Probability of correct prediction

class MailingListManager:
    """Manage email subscriptions"""
    def __init__(self, db_path: str = 'crypto_alerts.db'):
        self.db_path = db_path
        self.setup_database()
        self._subscribers_lock = threading.Lock()
        
    def setup_database(self):
        """Initialize the database tables"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS subscribers (
                    id INTEGER PRIMARY KEY,
                    email TEXT UNIQUE,
                    subscription_type TEXT,
                    active INTEGER DEFAULT 1,
                    created_at TEXT,
                    updated_at TEXT
                )
            ''')
    def add_subscriber(self, email: str, subscription_type: str = 'all') -> bool:
        """Add a new subscriber"""
        try:
            with self._subscribers_lock:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute('''
                        INSERT OR REPLACE INTO subscribers 
                        (email, subscription_type, created_at, updated_at)
                        VALUES (?, ?, datetime('now'), datetime('now'))
                    ''', (email, subscription_type))
            return True
        except Exception as e:
            print(f"Error adding subscriber: {e}")
            return False
            
    def remove_subscriber(self, email: str) -> bool:
        """Remove a subscriber"""
        try:
            with self._subscribers_lock:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute('''
                        UPDATE subscribers 
                        SET active = 0, updated_at = datetime('now')
                        WHERE email = ?
                    ''', (email,))
            return True
        except Exception as e:
            print(f"Error removing subscriber: {e}")
            return False
            
    def get_active_subscribers(self, subscription_type: Optional[str] = None) -> List[str]:
        """Get list of active subscribers"""
        with sqlite3.connect(self.db_path) as conn:
            query = '''
                SELECT email FROM subscribers 
                WHERE active = 1
            '''
            if subscription_type:
                query += ' AND subscription_type = ?'
                cursor = conn.execute(query, (subscription_type,))
            else:
                cursor = conn.execute(query)
                
            return [row[0] for row in cursor.fetchall()]

class EnhancedCryptoAlertSystem:
    """
    Main system class for cryptocurrency monitoring and alerts.
    Handles data fetching, analysis, and notifications.
    """
    
    def __init__(self, email_config: Dict, initial_recipients: List[str]):
        """Initialize the crypto alert system with email configuration."""
        self.email_config = email_config
        self.logger = self._setup_logging()
        self.running = True
        self.prediction_tracker = PredictionTracker()
        self.mailing_list = MailingListManager()

        # Add initial recipients to mailing list
        for email in initial_recipients:
            self.mailing_list.add_subscriber(email)

        # API rate limiting
        self.last_api_call = 0
        self.API_CALL_DELAY = 1.5  # seconds

        # Technical analysis parameters
        self.RSI_PERIOD = 14
        self.MACD_FAST = 12
        self.MACD_SLOW = 26
        self.MACD_SIGNAL = 9
        self.VOLUME_THRESHOLD = 2.0

        # BTC specific parameters
        self.BTC_CORRELATION_PERIOD = 30  # days
        self.BTC_INFLUENCE_THRESHOLD = 0.7

        # Market cap range
        self.MIN_MARKET_CAP = 5_000_000  # $5M
        self.MAX_MARKET_CAP = 100_000_000  # $100M

    def _setup_logging(self) -> logging.Logger:
        """Configure rotating log file handler."""
        logger = logging.getLogger('CryptoAlertSystem')
        logger.setLevel(logging.INFO)
        handler = logging.handlers.RotatingFileHandler(
            'crypto_alerts.log',
            maxBytes=10485760,  # 10MB
            backupCount=5
        )
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def wait_for_rate_limit(self):
        """Implement API rate limiting with longer delays"""
        current_time = time.time()
        time_since_last_call = current_time - self.last_api_call
        
        if time_since_last_call < self.API_CALL_DELAY:
            wait_time = self.API_CALL_DELAY - time_since_last_call + 1
            print(f"Rate limiting: Waiting {wait_time:.1f} seconds...")
            time.sleep(wait_time)
        
        self.last_api_call = time.time()

    def fetch_btc_data(self) -> Dict:
        """Fetch comprehensive Bitcoin data with enhanced retry mechanism"""
        MAX_RETRIES = 3
        BASE_DELAY = 15  # Increased base delay
        
        def make_request_with_retry(endpoint: str, params: Dict = None) -> Dict:
            for attempt in range(MAX_RETRIES):
                try:
                    if attempt > 0:
                        delay = BASE_DELAY * (2 ** attempt)
                        print(f"Waiting {delay} seconds before retry...")
                        time.sleep(delay)
                    
                    url = f"https://api.coingecko.com/api/v3{endpoint}"
                    print(f"Making request to: {url}")
                    self.wait_for_rate_limit()
                    response = requests.get(url, params=params)
                    
                    print(f"Response status code: {response.status_code}")
                    
                    if response.status_code == 429:
                        print(f"Rate limit hit, attempt {attempt + 1}/{MAX_RETRIES}")
                        if attempt == MAX_RETRIES - 1:
                            return {'error': 'rate_limit'}
                        continue
                    elif response.status_code != 200:
                        print(f"Error response: {response.text}")
                        return {'error': 'api_error'}
                        
                    data = response.json()
                    if data is None:
                        print(f"Warning: Received null response from {endpoint}")
                        return {'error': 'null_response'}
                        
                    return data
                    
                except Exception as e:
                    print(f"Request failed, attempt {attempt + 1}/{MAX_RETRIES}: {str(e)}")
                    if attempt == MAX_RETRIES - 1:
                        return {'error': 'request_failed'}
        
        try:
            print("\nFetching BTC data...")
            
            time.sleep(5)  
            
            btc_data = make_request_with_retry("/coins/bitcoin")
            if not btc_data or 'error' in btc_data:
                print("Failed to fetch BTC data")
                btc_data = {
                    'market_data': {
                        'current_price': {'usd': 0},
                        'price_change_percentage_24h': 0,
                        'price_change_percentage_7d': 0,
                        'market_cap_percentage': {'btc': 0}
                    }
                }
            print("BTC data fetched successfully")
            
            time.sleep(5)
            
            global_data = make_request_with_retry("/global")
            if not global_data or 'error' in global_data:
                print("Failed to fetch global data")
                global_data = {'data': {'market_cap_percentage': {'btc': 40.0}}}
            print("Global data fetched successfully")
            
            time.sleep(5)
            
            history_data = make_request_with_retry(
                "/coins/bitcoin/market_chart",
                params={'vs_currency': 'usd', 'days': 30, 'interval': 'daily'}
            )
            if not history_data or 'error' in history_data:
                print("Failed to fetch history data")
                current_price = btc_data.get('market_data', {}).get('current_price', {}).get('usd', 0)
                history_data = {
                    'prices': [[time.time() * 1000, current_price] for _ in range(30)]
                }
            print("Historical data fetched successfully")
            
            return {
                'current_data': btc_data,
                'global_data': global_data,
                'history': history_data
            }
            
        except Exception as e:
            self.logger.error(f"Error fetching BTC data: {str(e)}")
            print(f"Error: {str(e)}")
            print(traceback.format_exc())
            return {
                'current_data': {
                    'market_data': {
                        'current_price': {'usd': 0},
                        'price_change_percentage_24h': 0,
                        'price_change_percentage_7d': 0,
                        'market_cap_percentage': {'btc': 0}
                    }
                },
                'global_data': {'data': {'market_cap_percentage': {'btc': 40.0}}},
                'history': {'prices': [[time.time() * 1000, 0] for _ in range(30)]}
            }
    
    def fetch_crypto_data(self) -> pd.DataFrame:
        """Fetch cryptocurrency market data"""
        try:
            base_url = "https://api.coingecko.com/api/v3/coins/markets"
            all_data = []
            page = 1
            max_retries = 3

            while page <= 10:  # Fetch up to 10 pages
                print(f"\nFetching page {page}...")
                
                params = {
                    'vs_currency': 'usd',
                    'order': 'market_cap_desc',
                    'per_page': 250,
                    'page': page,
                    'sparkline': True,
                    'price_change_percentage': '24h,7d'
                }
                
                time.sleep(6.1)  
                
                for retry in range(max_retries):
                    try:
                        response = requests.get(base_url, params=params)
                        
                        if response.status_code == 200:
                            page_data = response.json()
                            if not page_data:
                                return pd.DataFrame(all_data)
                            
                            processed_data = []
                            for coin in page_data:
                                if coin.get('market_cap', 0) is not None and 5_000_000 <= coin['market_cap'] <= 100_000_000:
                                    processed_coin = {
                                        'id': coin.get('id', 'unknown'),
                                        'symbol': coin.get('symbol', 'unknown'),
                                        'name': coin.get('name', 'Unknown'),
                                        'current_price': coin.get('current_price', 0),
                                        'market_cap': coin.get('market_cap', 0),
                                        'total_volume': coin.get('total_volume', 0),
                                        'price_change_percentage_24h': coin.get('price_change_percentage_24h', 0),
                                        'price_change_percentage_7d': coin.get('price_change_percentage_7d', 0),
                                        'sparkline_in_7d': coin.get('sparkline_in_7d', {'price': [], 'volume': []})
                                    }
                                    processed_data.append(processed_coin)
                            
                            if processed_data:
                                all_data.extend(processed_data)
                                print(f"Found {len(processed_data)} coins in target range")
                            
                            page += 1
                            break
                            
                        elif response.status_code == 429:
                            if retry < max_retries - 1:
                                wait_time = (retry + 1) * 60
                                print(f"Rate limit hit, waiting {wait_time} seconds...")
                                time.sleep(wait_time)
                            else:
                                print("Rate limit exceeded, returning current data")
                                return pd.DataFrame(all_data)
                        else:
                            print(f"Error: Status code {response.status_code}")
                            return pd.DataFrame(all_data)
                            
                    except Exception as e:
                        print(f"Error on retry {retry + 1}: {str(e)}")
                        if retry == max_retries - 1:
                            return pd.DataFrame(all_data)
                        time.sleep(10)

            return pd.DataFrame(all_data)
            
        except Exception as e:
            self.logger.error(f"Error fetching crypto data: {str(e)}")
            return pd.DataFrame()

    def check_market_conditions(self):
        """Monitor market conditions without sending alerts"""
        try:
            print("\nChecking market conditions...")
            btc_analysis = self.analyze_btc()
            df = self.fetch_crypto_data()
            
            if btc_analysis.price == 0:
                return
                
            # Log significant movements for weekly report
            if abs(btc_analysis.change_24h) >= 10:
                print(f"Significant BTC movement: {btc_analysis.change_24h:+.2f}% (logged for weekly report)")
            
            # Log significant altcoin movements
            if not df.empty:
                significant_movers = df[
                    (abs(df['price_change_percentage_24h']) > 30) &
                    (df['market_cap'] >= 10_000_000) &
                    (df['total_volume'] >= 1_000_000)
                ]
                
                if not significant_movers.empty:
                    print(f"Found {len(significant_movers)} significant altcoin movements (logged for weekly report)")
            
            print("Market check completed - Data stored for weekly report")
        
        except Exception as e:
            self.logger.error(f"Error checking market conditions: {str(e)}")
            print(f"Error in market check: {str(e)}")

    def analyze_btc(self) -> BTCAnalysis:
        """Comprehensive Bitcoin analysis with prediction learning"""
        try:
            print("Starting BTC analysis...")
            btc_data = self.fetch_btc_data()
            
            # Extract current price and basic metrics
            current_data = btc_data.get('current_data', {})
            market_data = current_data.get('market_data', {})
            current_price = market_data.get('current_price', {}).get('usd', 0)
            change_24h = market_data.get('price_change_percentage_24h', 0)
            change_7d = market_data.get('price_change_percentage_7d', 0)
            
            # Get price history
            history_data = btc_data.get('history', {})
            price_history = [p[1] for p in history_data.get('prices', [])]
            if not price_history:
                price_history = [current_price]
            
            #market dominance
            dominance = btc_data.get('global_data', {}).get('data', {}).get('market_cap_percentage', {}).get('btc', 40.0)
            
            
            np_prices = np.array(price_history, dtype=float)
            if len(np_prices) > self.RSI_PERIOD:
                rsi = talib.RSI(np_prices)[-1]
                macd, signal, _ = talib.MACD(
                    np_prices,
                    fastperiod=self.MACD_FAST,
                    slowperiod=self.MACD_SLOW,
                    signalperiod=self.MACD_SIGNAL
                )
                bb_upper, bb_middle, bb_lower = talib.BBANDS(np_prices)
            else:
                rsi = 50
                macd = signal = np.array([0])
                bb_upper = bb_middle = bb_lower = np.array([current_price])
            
           
            levels = self.identify_support_resistance(np_prices)
            
            
            correlations = self.calculate_market_correlations(price_history)
            
            
            sentiment = self.calculate_btc_sentiment(
                btc_data, 
                rsi, 
                macd[-1] if not np.isnan(macd[-1]) else 0, 
                signal[-1] if not np.isnan(signal[-1]) else 0
            )
            
            
            analysis = BTCAnalysis(
                price=current_price,
                price_history=price_history,
                change_24h=change_24h,
                change_7d=change_7d,
                dominance=dominance,
                correlation=correlations,
                technical_indicators={
                    'rsi': rsi,
                    'macd': macd[-1] if not np.isnan(macd[-1]) else 0,
                    'macd_signal': signal[-1] if not np.isnan(signal[-1]) else 0,
                    'bb_upper': bb_upper[-1] if not np.isnan(bb_upper[-1]) else current_price * 1.1,
                    'bb_lower': bb_lower[-1] if not np.isnan(bb_lower[-1]) else current_price * 0.9
                },
                market_sentiment=sentiment,
                support_levels=levels['support'],
                resistance_levels=levels['resistance']
            )

            
            features = {
                'rsi': analysis.technical_indicators['rsi'],
                'macd': analysis.technical_indicators['macd'],
                'dominance': analysis.dominance,
                'price_change_24h': analysis.change_24h,
                'price_change_7d': analysis.change_7d
            }
            
            #prediction confidence
            confidence = self.prediction_tracker.get_prediction_confidence(features)
            
            #Record prediction for future learning
            prediction_record = PredictionRecord(
                timestamp=datetime.now().isoformat(),
                coin_symbol='BTC',
                prediction_type='price_movement',
                predicted_value=analysis.price * (1 + analysis.change_24h/100),
                actual_value=0,  
                features=features,
                was_correct=False  
            )
            self.prediction_tracker.record_prediction(prediction_record)
            
            
            analysis.prediction_confidence = confidence
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error in BTC analysis: {str(e)}")
            print(traceback.format_exc())
            return BTCAnalysis(
                price=0.0,
                price_history=[],
                change_24h=0.0,
                change_7d=0.0,
                dominance=0.0,
                correlation={},
                technical_indicators={
                    'rsi': 0,
                    'macd': 0,
                    'macd_signal': 0,
                    'bb_upper': 0,
                    'bb_lower': 0
                },
                market_sentiment="Neutral",
                support_levels=[],
                resistance_levels=[],
                prediction_confidence=0.0
            )

    def identify_support_resistance(self, prices: np.array) -> Dict[str, List[float]]:
        """Identify key support and resistance levels"""
        window_size = 5
        support = []
        resistance = []
        
        for i in range(window_size, len(prices) - window_size):
            if all(prices[i] <= prices[i-window_size:i]) and all(prices[i] <= prices[i+1:i+window_size+1]):
                support.append(prices[i])
            if all(prices[i] >= prices[i-window_size:i]) and all(prices[i] >= prices[i+1:i+window_size+1]):
                resistance.append(prices[i])
        
        return {
            'support': sorted(set(support))[-3:],
            'resistance': sorted(set(resistance))[:3]
        }

    def calculate_btc_sentiment(self, btc_data: Dict, rsi: float, macd: float, signal: float) -> str:
        """Calculate overall BTC market sentiment with detailed analysis"""
        try:
            sentiment_score = 0
            analysis_points = []
            
            print("\n=== DETAILED MARKET ANALYSIS ===")
            
            # RSI Analysis
            if rsi > 75:
                sentiment_score -= 3
                analysis_points.append(f"🔴 Extremely overbought (RSI: {rsi:.2f})")
            elif rsi > 70:
                sentiment_score -= 2
                analysis_points.append(f"🟡 Strongly overbought (RSI: {rsi:.2f})")
            elif rsi > 60:
                sentiment_score -= 1
                analysis_points.append(f"⚪ Approaching overbought (RSI: {rsi:.2f})")
            elif rsi < 30:
                sentiment_score += 2
                analysis_points.append(f"🟢 Oversold (RSI: {rsi:.2f})")
            
            # Price Action Analysis
            current_data = btc_data.get('current_data', {})
            market_data = current_data.get('market_data', {})
            
            price_change_24h = market_data.get('price_change_percentage_24h', 0)
            price_change_7d = market_data.get('price_change_percentage_7d', 0)
            
            analysis_points.append(f"24h Change: {price_change_24h:+.2f}%")
            analysis_points.append(f"7d Change: {price_change_7d:+.2f}%")
            
            if price_change_24h > 5:
                sentiment_score += 2
                analysis_points.append("🟢 Strong 24h momentum")
            elif price_change_24h < -5:
                sentiment_score -= 2
                analysis_points.append("🔴 Weak 24h momentum")
                
            if price_change_7d > 10:
                sentiment_score += 1
                analysis_points.append("🟢 Positive weekly trend")
            elif price_change_7d < -10:
                sentiment_score -= 1
                analysis_points.append("🔴 Negative weekly trend")
            
            # Volume Analysis
            current_volume = market_data.get('total_volume', {}).get('usd', 0)
            market_cap = market_data.get('market_cap', {}).get('usd', 1)
            
            if market_cap > 0:
                volume_to_mcap = (current_volume / market_cap) * 100
                analysis_points.append(f"Volume/Market Cap Ratio: {volume_to_mcap:.2f}%")
                
                if volume_to_mcap > 10:
                    sentiment_score += 2
                    analysis_points.append("🟢 Exceptional trading volume")
                elif volume_to_mcap > 5:
                    sentiment_score += 1
                    analysis_points.append("🟢 Strong trading volume")
                elif volume_to_mcap < 2:
                    sentiment_score -= 1
                    analysis_points.append("🔴 Low trading volume")
            
            # Market Dominance Analysis
            global_data = btc_data.get('global_data', {})
            if isinstance(global_data, dict):
                market_data = global_data.get('data', {})
                if isinstance(market_data, dict):
                    dominance = market_data.get('market_cap_percentage', {}).get('btc', 40.0)
                    analysis_points.append(f"BTC Dominance: {dominance:.2f}%")
                    
                    if dominance > 55:
                        sentiment_score += 2
                        analysis_points.append("🟢 Very strong market dominance")
                    elif dominance > 50:
                        sentiment_score += 1
                        analysis_points.append("🟢 Above average market dominance")
                    elif dominance < 45:
                        sentiment_score -= 1
                        analysis_points.append("🔴 Below average market dominance")
            
            # MACD Analysis
            if macd > signal:
                sentiment_score += 1
                analysis_points.append("🟢 MACD above signal line")
            elif macd < signal:
                sentiment_score -= 1
                analysis_points.append("🔴 MACD below signal line")
            
            # Print analysis points
            for point in analysis_points:
                print(point)
            
            print(f"\n📊 Final Sentiment Score: {sentiment_score}")
            
            # Convert score to sentiment
            if sentiment_score >= 5:
                return "Strongly Bullish"
            elif sentiment_score >= 2:
                return "Bullish"
            elif sentiment_score >= -1:
                return "Neutral"
            elif sentiment_score >= -4:
                return "Bearish"
            else:
                return "Strongly Bearish"
                
        except Exception as e:
            print(f"Error in sentiment calculation: {str(e)}")
            print("Detailed error:")
            traceback.print_exc()
            return "Neutral"

    def calculate_market_correlations(self, btc_prices: List[float]) -> Dict[str, float]:
        """Calculate correlations between BTC and other market data"""
        try:
            # Fetch comparison data
            sp500_data = self.fetch_market_data('SP500')
            gold_data = self.fetch_market_data('GOLD')
            
            correlations = {}
            btc_returns = np.diff(btc_prices) / btc_prices[:-1]
            
            # Calculate correlations if data is available
            if sp500_data:
                sp500_returns = np.diff(sp500_data) / sp500_data[:-1]
                correlations['SP500'] = np.corrcoef(btc_returns, sp500_returns)[0, 1]
                
            if gold_data:
                gold_returns = np.diff(gold_data) / gold_data[:-1]
                correlations['GOLD'] = np.corrcoef(btc_returns, gold_returns)[0, 1]
            
            return correlations
            
        except Exception as e:
            self.logger.error(f"Error calculating correlations: {str(e)}")
            return {}

    def fetch_market_data(self, asset: str) -> List[float]:
        """Fetch market data for comparison assets"""
        try:
            base_url = "https://www.alphavantage.co/query"
            api_key = "YOUR_ALPHA_VANTAGE_KEY"  # You can get a free key from Alpha Vantage
            
            if asset == 'SP500':
                symbol = 'SPY'
            elif asset == 'GOLD':
                symbol = 'GLD'
            else:
                return []
            
            params = {
                'function': 'TIME_SERIES_DAILY',
                'symbol': symbol,
                'apikey': api_key
            }
            
            response = requests.get(base_url, params=params)
            data = response.json()
            
            time_series = data.get('Time Series (Daily)', {})
            prices = [float(v['4. close']) for v in list(time_series.values())[:30]]
            return prices[::-1]  # Reverse to get chronological order
            
        except Exception as e:
            self.logger.error(f"Error fetching market data for {asset}: {str(e)}")
            return []

    def send_email(self, subject: str, body: str):
        """Send email to all active subscribers"""
        try:
            subscribers = self.mailing_list.get_active_subscribers()
            
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.email_config['sender']
            msg['To'] = ", ".join(subscribers)
            
            msg.attach(MIMEText(body, 'html'))
            
            with smtplib.SMTP_SSL(
                self.email_config['smtp_server'],
                self.email_config['port']
            ) as server:
                server.login(
                    self.email_config['username'],
                    self.email_config['password']
                )
                server.send_message(msg)
                
            self.logger.info(f"Email sent successfully to {len(subscribers)} subscribers")
            
        except Exception as e:
            self.logger.error(f"Error sending email: {str(e)}")

    def send_weekly_report(self):
        """Generate and send weekly report"""
        try:
            report = self.generate_weekly_report()
            self.send_email(
                subject=f"Weekly Crypto Analysis Report - {datetime.now().strftime('%Y-%m-%d')}",
                body=report
            )
            self.logger.info("Weekly report sent successfully")
        except Exception as e:
            self.logger.error(f"Error sending weekly report: {str(e)}")

    def update_historical_predictions(self):
        """Update past predictions with actual outcomes"""
        try:
            
            btc_data = self.fetch_btc_data()
            current_price = btc_data.get('current_data', {}).get('market_data', {}).get('current_price', {}).get('usd', 0)

            # Update predictions from 24 hours ago
            timestamp = (datetime.now() - timedelta(hours=24)).isoformat()
            self.prediction_tracker.update_prediction_outcome(
                timestamp, 'BTC', current_price
            )
            
            
            self.prediction_tracker.train_model()
            
        except Exception as e:
            self.logger.error(f"Error updating predictions: {str(e)}")

    def generate_weekly_report(self) -> str:
        """Generate enhanced weekly report including prediction confidence"""
        try:
            df = self.fetch_crypto_data()
            btc_analysis = self.analyze_btc()
            
            if df is None or df.empty:
                return """
                    <html><body><h1>Crypto Market Report</h1>
                    <p>Unable to fetch market data at this time. Please try again later.</p>
                    </body></html>
                """
            
            
            required_columns = ['market_cap', 'price_change_percentage_7d', 'current_price', 'symbol', 'name']
            for col in required_columns:
                if col not in df.columns:
                    df[col] = 0 if col != 'symbol' and col != 'name' else 'unknown'
                df[col] = df[col].fillna(0 if col != 'symbol' and col != 'name' else 'unknown')

            report = f"""
                <html>
                <head>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
                        .container {{ max-width: 800px; margin: 0 auto; }}
                        .header {{ background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin-bottom: 20px; }}
                        .section {{ margin-bottom: 30px; }}
                        .coin-card {{ 
                            border: 1px solid #ddd; 
                            border-radius: 10px; 
                            padding: 20px; 
                            margin-bottom: 20px;
                            background-color: white;
                        }}
                        .metrics-grid {{ 
                            display: grid; 
                            grid-template-columns: 1fr 1fr; 
                            gap: 15px; 
                            margin-bottom: 15px; 
                        }}
                        .price-info, .volume-info {{ padding: 10px; }}
                        .positive {{ color: #28a745; }}
                        .negative {{ color: #dc3545; }}
                        .confidence-meter {{
                            background-color: #f8f9fa;
                            padding: 10px;
                            border-radius: 5px;
                            margin-top: 10px;
                        }}
                        .confidence-bar {{
                            height: 20px;
                            background: linear-gradient(to right, #dc3545 0%, #ffc107 50%, #28a745 100%);
                            border-radius: 10px;
                            position: relative;
                        }}
                        .confidence-marker {{
                            position: absolute;
                            width: 2px;
                            height: 30px;
                            background-color: black;
                            bottom: -5px;
                        }}
                        .risk-low {{ color: #28a745; }}
                        .risk-medium {{ color: #ffc107; }}
                        .risk-high {{ color: #dc3545; }}
                        .opportunity-section {{
                            background-color: #f8f9fa;
                            padding: 15px;
                            border-radius: 5px;
                            margin-top: 15px;
                        }}
                        .disclaimer {{
                            font-size: 0.9em;
                            color: #666;
                            text-align: center;
                            margin-top: 30px;
                            padding: 15px;
                            background-color: #f8f9fa;
                            border-radius: 5px;
                        }}
                    </style>
                </head>
                <body>
                    <div class="container">
                        <div class="header">
                            <h1>Weekly Crypto Market Report</h1>
                            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
                        </div>

                        <div class="section">
                            <h2>Bitcoin Analysis</h2>
                            <div style="padding: 15px; background-color: white; border-radius: 10px; border: 1px solid #ddd;">
                                <div style="font-size: 18px; margin: 10px 0;">Price: ${btc_analysis.price:,.2f}</div>
                                <div class="{('positive' if btc_analysis.change_24h > 0 else 'negative')}" style="font-size: 18px; margin: 10px 0;">
                                    24h Change: {btc_analysis.change_24h:+.2f}%
                                </div>
                                <div class="{('positive' if btc_analysis.change_7d > 0 else 'negative')}" style="font-size: 18px; margin: 10px 0;">
                                    7d Change: {btc_analysis.change_7d:+.2f}%
                                </div>
                                <div style="font-size: 18px; margin: 10px 0;">Dominance: {btc_analysis.dominance:.2f}%</div>
                                <div style="font-size: 18px; margin: 10px 0;">Market Sentiment: {btc_analysis.market_sentiment}</div>
                                
                                <div class="confidence-meter">
                                    <h4>Prediction Confidence: {btc_analysis.prediction_confidence * 100:.1f}%</h4>
                                    <div class="confidence-bar">
                                        <div class="confidence-marker" style="left: {btc_analysis.prediction_confidence * 100}%;"></div>
                                    </div>
                                    <p style="font-size: 0.9em; color: #666;">
                                        Based on historical prediction accuracy
                                    </p>
                                </div>
                            </div>
                        </div>
            """
            
            try:
                small_caps = df[
                    (df['market_cap'].notna()) & 
                    (df['market_cap'] >= self.MIN_MARKET_CAP) & 
                    (df['market_cap'] <= self.MAX_MARKET_CAP)
                ].copy()
                
                small_caps['price_change_percentage_7d'] = pd.to_numeric(
                    small_caps['price_change_percentage_7d'],
                    errors='coerce'
                ).fillna(0)
                
                #top performers
                top_performers = small_caps.nlargest(5, 'price_change_percentage_7d')
                
                if not top_performers.empty:
                    report += """
                        <div class="section">
                            <h2>📈 Small-Cap Opportunities</h2>
                            <p>Top performers in the $5M-$100M market cap range:</p>
                    """
                    
                    for _, coin in top_performers.iterrows():
                        report += f"""
                            <div class="coin-card">
                                <h3>{coin['name']} ({coin['symbol'].upper()})</h3>
                                <div class="metrics-grid">
                                    <div class="price-info">
                                        <p>Price: ${coin['current_price']:,.6f}</p>
                                        <p>Market Cap: ${coin['market_cap']:,.0f}</p>
                                        <p>7d Change: <span class="{('positive' if coin['price_change_percentage_7d'] > 0 else 'negative')}">{coin['price_change_percentage_7d']:+.2f}%</span></p>
                                    </div>
                                    <div class="volume-info">
                                        <p>Volume: ${coin['total_volume']:,.0f}</p>
                                        <p>Volume/MCap: {(coin['total_volume'] / coin['market_cap'] * 100):.1f}%</p>
                                    </div>
                                </div>
                            </div>
                        """
                        
            except Exception as e:
                self.logger.error(f"Error processing small caps: {str(e)}")
            
            # Add disclaimer and close HTML
            report += """
                        </div>
                        <div class="disclaimer">
                            <p>This report focuses on cryptocurrencies with market caps between $5M and $100M USD.</p>
                            <p>All investments carry risk. Always conduct your own research before making investment decisions.</p>
                            <p>Past performance does not guarantee future results.</p>
                        </div>
                    </div>
                </body>
                </html>
            """
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating weekly report: {str(e)}")
            return f"""
                <html>
                <body>
                    <h1>Error Generating Report</h1>
                    <p>An error occurred while generating the report: {str(e)}</p>
                    <p>Please try again later.</p>
                </body>
                </html>
            """

    def run_scheduled_tasks(self):
        """Configure and run scheduled tasks with enhanced logging"""
        try:
            print("\n" + "="*50)
            print("CRYPTO ALERT SYSTEM INITIALIZATION")
            print("="*50)
            print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("\nEmail Schedule:")
            print("Weekly Report Only:")
            print("- Every Friday at 16:00")
            print("- Full market analysis and opportunities")
            
            # Configure schedules
            schedule.clear()  # Clear any existing schedules
            
            # Calculate next weekly report time
            next_friday = datetime.now()
            while next_friday.weekday() != 4:  # 4 is Friday
                next_friday += timedelta(days=1)
            next_friday = next_friday.replace(hour=16, minute=0, second=0, microsecond=0)
            if next_friday < datetime.now():
                next_friday += timedelta(days=7)
            
            # Schedule weekly reports for Friday
            schedule.every().friday.at("16:00").do(self.send_weekly_report)
            print(f"\n✓ Weekly report scheduled - Next report: {next_friday.strftime('%Y-%m-%d %H:%M:%S')}")
            time_to_report = next_friday - datetime.now()
            print(f"  ({int(time_to_report.total_seconds()/3600)} hours from now)")
            
            # Schedule hourly checks (for data collection only)
            schedule.every().hour.do(self.check_market_conditions)
            print("\n✓ Market monitoring active (no alerts, data collection only)")
            
            print("\nRunning initial market check...")
            sys.stdout.flush()
            
            # Run initial market check
            try:
                self.check_market_conditions()
                print("Initial market check completed successfully")
            except Exception as e:
                print(f"Error in initial market check: {str(e)}")
                print(traceback.format_exc())
            
            print("\nSystem is now running - Emails only on Fridays at 16:00")
            print("="*50)
            
            # Main loop
            while self.running:
                try:
                    schedule.run_pending()
                    time.sleep(60)  # Check every minute
                    
                    # Print hourly status update
                    current_time = datetime.now()
                    if current_time.minute == 0:
                        print(f"\nStatus Update ({current_time.strftime('%Y-%m-%d %H:%M:%S')})")
                        print("-" * 30)
                        
                        # Time until next weekly report
                        next_report = next_friday
                        while next_report < current_time:
                            next_report += timedelta(days=7)
                        time_to_report = next_report - current_time
                        
                        print(f"Next weekly report: {next_report.strftime('%Y-%m-%d %H:%M:%S')}")
                        print(f"Time until report: {int(time_to_report.total_seconds()/3600)} hours")
                        sys.stdout.flush()
                    
                except Exception as e:
                    print(f"Error in main loop: {str(e)}")
                    print(traceback.format_exc())
                    time.sleep(60)
                    
        except Exception as e:
            self.logger.error(f"Error in scheduling tasks: {str(e)}")
            print(f"Critical Error: {str(e)}")
            print(traceback.format_exc())
            raise

def setup_background_service():
    try:
        working_dir = os.getcwd()
        script_path = os.path.join(working_dir, 'crypto_alert_system.py')
        
        plist_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.crypto.alert</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/bin/python3</string>
        <string>{script_path}</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardErrorPath</key>
    <string>{working_dir}/crypto_alert_error.log</string>
    <key>StandardOutPath</key>
    <string>{working_dir}/crypto_alert.log</string>
    <key>WorkingDirectory</key>
    <string>{working_dir}</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
        <key>PYTHONPATH</key>
        <string>{working_dir}</string>
    </dict>
</dict>
</plist>
"""
        
        # Create LaunchAgent directory if it doesn't exist
        launch_agents_dir = os.path.expanduser('~/Library/LaunchAgents')
        os.makedirs(launch_agents_dir, exist_ok=True)
        
        # Write plist file
        plist_path = os.path.join(launch_agents_dir, 'com.crypto.alert.plist')
        with open(plist_path, 'w') as f:
            f.write(plist_content)
        
        # Set correct permissions
        os.chmod(plist_path, 0o644)
        
        # Load the service
        os.system(f'launchctl unload {plist_path} 2>/dev/null')
        os.system(f'launchctl load {plist_path}')
        
        print("\nService successfully installed!")
        print("\nUseful commands:")
        print("- Start service: launchctl start com.crypto.alert")
        print("- Stop service: launchctl stop com.crypto.alert")
        print("- View logs: check crypto_alert.log and crypto_alert_error.log")
        
    except Exception as e:
        print(f"Error setting up service: {str(e)}")
        print("Please ensure you have the necessary permissions and try again")

def install_required_packages():
    """Install required packages on macOS"""
    try:
        print("Installing required packages...")
        os.system("pip3 install pandas numpy requests schedule scikit-learn")
        os.system("brew install ta-lib")
        os.system("pip3 install TA-Lib")
        print("Packages installed successfully!")
    except Exception as e:
        print(f"Error installing packages: {str(e)}")

if __name__ == "__main__":
    
    email_config = {
       'sender': 'boatengshadrack27@gmail.com',
       'smtp_server': 'smtp.gmail.com',
       'port': 465,
       'username': 'boatengshadrack27@gmail.com',
       'password': 'wtfe nsma iiiq ojsi'
    }

    initial_recipients = ['boatengshadrack27@gmail.com', 'mannymart026@gmail.com', 'Reyes.maru23@hotmail.com']

    if len(sys.argv) > 1:
        if sys.argv[1] == '--setup-service':
            setup_background_service()
        elif sys.argv[1] == '--install-packages':
            install_required_packages()
        elif sys.argv[1] == '--add-subscriber':
            if len(sys.argv) > 2:
                try:
                    alert_system = EnhancedCryptoAlertSystem(email_config, [])
                    if alert_system.mailing_list.add_subscriber(sys.argv[2]):
                        print(f"Successfully added subscriber: {sys.argv[2]}")
                    else:
                        print("Failed to add subscriber")
                except Exception as e:
                    print(f"Error adding subscriber: {e}")
            else:
                print("Please provide an email address")
        elif sys.argv[1] == '--remove-subscriber':
            if len(sys.argv) > 2:
                try:
                    alert_system = EnhancedCryptoAlertSystem(email_config, [])
                    if alert_system.mailing_list.remove_subscriber(sys.argv[2]):
                        print(f"Successfully removed subscriber: {sys.argv[2]}")
                    else:
                        print("Failed to remove subscriber")
                except Exception as e:
                    print(f"Error removing subscriber: {e}")
            else:
                print("Please provide an email address")
        elif sys.argv[1] == '--list-subscribers':
            try:
                alert_system = EnhancedCryptoAlertSystem(email_config, [])
                subscribers = alert_system.mailing_list.get_active_subscribers()
                print("\nActive subscribers:")
                for email in subscribers:
                    print(f"- {email}")
            except Exception as e:
                print(f"Error listing subscribers: {e}")
        elif sys.argv[1] == '--test-report':
            try:
                print("Testing weekly report generation...")
                alert_system = EnhancedCryptoAlertSystem(email_config, initial_recipients)
                report = alert_system.generate_weekly_report()
                alert_system.send_email(
                    subject="Crypto Weekly Report - Test",
                    body=report
                )
                print("Test report generated and sent successfully!")
            except Exception as e:
                print(f"Error testing report: {str(e)}")
                traceback.print_exc()
        elif sys.argv[1] == '--test-email':
            try:
                alert_system = EnhancedCryptoAlertSystem(email_config, initial_recipients)
                alert_system.send_email(
                    subject="Crypto Alert System - Test Email",
                    body="<html><body><h2>Test Email</h2><p>System is working!</p></body></html>"
                )
                print("Test email sent successfully!")
            except Exception as e:
                print(f"Error sending test email: {str(e)}")
        elif sys.argv[1] == '--run':
            try:
                print("Starting Crypto Alert System...")
                alert_system = EnhancedCryptoAlertSystem(email_config, initial_recipients)
                alert_system.run_scheduled_tasks()
            except KeyboardInterrupt:
                print("\nShutting down gracefully...")
            except Exception as e:
                print(f"Error running alert system: {str(e)}")
                traceback.print_exc()
    else:
        try:
            print("Starting Crypto Alert System...")
            alert_system = EnhancedCryptoAlertSystem(email_config, initial_recipients)
            alert_system.run_scheduled_tasks()
        except KeyboardInterrupt:
            print("\nShutting down gracefully...")
        except Exception as e:
            print(f"Error running alert system: {str(e)}")
            traceback.print_exc()
    