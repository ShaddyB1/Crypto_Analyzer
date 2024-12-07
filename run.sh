#!/bin/bash

# Change to the script directory
cd "$(dirname "$0")"

# Run the Python script
/opt/anaconda3/bin/python3 crypto_alert_system.py --run >> crypto_alert.log 2>> crypto_alert_error.log