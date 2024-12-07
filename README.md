# Crypto Market Analyzer

A Python-based cryptocurrency market analysis system that:
- Monitors Bitcoin and altcoin markets
- Sends weekly analysis reports every Friday
- Tracks price movements, market sentiment, and technical indicators
- Uses machine learning to improve prediction accuracy over time

## Requirements
- Python 3.x
- Required packages: pandas, numpy, requests, schedule, scikit-learn, TA-Lib

## Installation
```bash
python3 crypto_alert_system.py --install-packages
```

## Usage
```bash
# Run the system
python3 crypto_alert_system.py --run

# Test email functionality
python3 crypto_alert_system.py --test-email

# Test report generation
python3 crypto_alert_system.py --test-report
```
