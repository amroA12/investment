# AI Quantitative Trading System

Author: Amro Elzaygh

## Overview

This project implements a quantitative trading strategy using machine learning.

The system predicts the best assets to hold using an ensemble model built from:

1. Transformer Neural Network
2. Random Forest Classifier

Predictions from both models are averaged to produce the final portfolio selection.

---

## Data Source

Market data is downloaded from Yahoo Finance using the yfinance library.

Assets used in the model include major technology stocks and Sharia-compliant ETFs such as:

AAPL, MSFT, GOOGL, AMD, ADBE, CSCO, QCOM, TXN, INTU, CRM, AVGO, ORCL, META, TSLA, INTC, KLAC, ADI, SNPS, CDNS, FTNT, SPUS, HLAL

Start date: 2010

---

## Feature Engineering

The model uses multiple technical indicators including:

Momentum:
- 1 week return
- 1 month return
- 3 month return

Volatility:
- 1 month volatility
- 3 month volatility

Trend indicators:
- Moving average ratios (10/50)
- Moving average ratios (50/200)

Technical indicators:
- RSI
- MACD
- ATR
- Bollinger Bands

Volume:
- Volume change

All features are standardized using StandardScaler.

---

## Machine Learning Models

Two models are trained:

Transformer Neural Network
- Uses multi-head attention
- Processes sequential time-series features
- Lookback window: 42 days

Random Forest Classifier
- 400 trees
- Max depth = 8
- Balanced class weights

Predictions from both models are averaged to form the ensemble prediction.

---

## Training Method

Walk-forward training is used.

Train window: 1000 samples  
Test window: 200 samples

This simulates real-world trading conditions.

---

## Portfolio Construction

Each rebalance period:

1. The model predicts probabilities for all assets
2. The top N assets are selected (N = 5)
3. Portfolio weights are calculated using inverse volatility weighting

---

## Risk Management

A market regime filter is used:

If SPUS price is below its 200-day moving average the strategy moves to cash.

---

## Backtesting

Rebalance frequency: 5 days

Transaction cost: 0.3%

Performance metrics include:

- Cumulative return
- Sharpe ratio

The benchmark used is SPUS.

---

## Running the Project

Install dependencies:

pip install -r requirements.txt

Run the model:

python trading_model.py

---

## Tests

The project includes automated tests using pytest.

To run the tests:

pytest

The tests validate:

- Technical indicators
- Model output shapes
- Probability values