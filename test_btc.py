from crypto_alert_system import EnhancedCryptoAlertSystem
import requests
import time

def test_api_connection():
    """Test basic API connectivity"""
    try:
        response = requests.get('https://api.coingecko.com/api/v3/ping')
        print(f"API Response: {response.text}")
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            print("API connection successful!")
            return True
        else:
            print(f"API connection failed with status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"API connection error: {str(e)}")
        return False

def print_analysis(analysis):
    if analysis:
        print("\n" + "="*50)
        print("🔵 BITCOIN MARKET ANALYSIS")
        print("="*50)
        
        print("\n📊 CURRENT METRICS")
        print(f"Price: ${analysis.price:,.2f}")
        print(f"24h Change: {analysis.change_24h:+.2f}%")
        print(f"7d Change: {analysis.change_7d:+.2f}%")
        print(f"Market Dominance: {analysis.dominance:.2f}%")
        
        print("\n📈 TECHNICAL INDICATORS")
        indicators = analysis.technical_indicators
        print(f"RSI ({indicators['rsi']:.2f}): "
              f"{'Overbought' if indicators['rsi'] > 70 else 'Oversold' if indicators['rsi'] < 30 else 'Neutral'}")
        print(f"MACD: {indicators['macd']:.2f}")
        print(f"Signal: {indicators['macd_signal']:.2f}")
        print(f"Bollinger Bands: {indicators['bb_lower']:.2f} - {indicators['bb_upper']:.2f}")
        
        print("\n🎯 SUPPORT & RESISTANCE")
        print("Support Levels:", [f"${x:,.2f}" for x in analysis.support_levels])
        print("Resistance Levels:", [f"${x:,.2f}" for x in analysis.resistance_levels])
        
        if analysis.correlation:
            print("\n🔗 MARKET CORRELATIONS")
            for asset, corr in analysis.correlation.items():
                print(f"{asset}: {corr:+.2f}")
        
        print("\n🔮 MARKET SENTIMENT")
        sentiment_emoji = {
            "Strongly Bullish": "🟢",
            "Bullish": "🟡",
            "Neutral": "⚪",
            "Bearish": "🟡",
            "Strongly Bearish": "🔴"
        }
        print(f"{sentiment_emoji.get(analysis.market_sentiment, '⚪')} {analysis.market_sentiment}")
        
        print("\n" + "="*50)
    else:
        print("❌ Analysis failed!")


email_config = {
       'sender': 'your_email@gmail.com',
       'smtp_server': 'smtp.gmail.com',
       'port': 465,
       'username': 'your_email@gmail.com',
       'password': 'generated_pin'
    }

recipient_emails = ['boatengshadrack27@gmail.com', 'mannymart026@icloud.com']

print("Testing API connection...")
if not test_api_connection():
    print("Please check your internet connection")
    exit(1)

print("\nStarting BTC analysis test...")
system = EnhancedCryptoAlertSystem(email_config, recipient_emails)
analysis = system.analyze_btc()
print_analysis(analysis)
