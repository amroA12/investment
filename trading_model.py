import yfinance as yf
import pandas as pd
import numpy as np
import random
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

from keras.layers import (
    Input, Dense, Dropout,
    LayerNormalization,
    MultiHeadAttention,
    GlobalAveragePooling1D
)
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

import joblib
import warnings
warnings.filterwarnings("ignore")

# ==========================================
# Random Seed 
# ==========================================

seed = 42
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)

# ==========================================
# Configuration
# ==========================================

funds = [
    "AAPL","MSFT","GOOGL","AMD","ADBE",
    "CSCO","QCOM","TXN","INTU",
    "CRM","AVGO","ORCL","META","TSLA",
    "INTC","KLAC","ADI","SNPS","CDNS",
    "FTNT","SPUS","HLAL"
]

start_date = "2010-01-01"

lookback = 42
top_n = 5
rebalance_period = 5
transaction_cost = 0.003

train_window = 1000
test_window = 200

# ==========================================
# Download Market Data
# ==========================================

data = yf.download(funds, start=start_date)

close = data["Close"]
high = data["High"]
low = data["Low"]
volume = data["Volume"]

returns = close.pct_change()

# ==========================================
# Technical Indicators
# ==========================================

def compute_RSI(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss
    return 100 - (100/(1+rs))

def compute_MACD(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast).mean()
    ema_slow = series.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal).mean()
    return macd - signal_line

def compute_ATR(high, low, close, period=14):
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def bollinger(series, window=20):
    ma = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = ma + 2*std
    lower = ma - 2*std
    return upper, lower

# ==========================================
# Feature Engineering
# ==========================================

features = pd.DataFrame(index=close.index)

for fund in funds:

    price = close[fund]
    ret = returns[fund]

    ma10 = price.rolling(10).mean()
    ma20 = price.rolling(20).mean()
    ma50 = price.rolling(50).mean()
    ma200 = price.rolling(200).mean()

    bb_up, bb_low = bollinger(price)

    atr = compute_ATR(high[fund], low[fund], price)

    features[f"{fund}_ret_1w"] = ret.rolling(5).sum()
    features[f"{fund}_ret_1m"] = ret.rolling(21).sum()
    features[f"{fund}_ret_3m"] = ret.rolling(63).sum()

    features[f"{fund}_vol_1m"] = ret.rolling(21).std()
    features[f"{fund}_vol_3m"] = ret.rolling(63).std()

    features[f"{fund}_trend_10_50"] = ma10 / ma50
    features[f"{fund}_trend_50_200"] = ma50 / ma200

    features[f"{fund}_RSI"] = compute_RSI(price)
    features[f"{fund}_MACD"] = compute_MACD(price)

    features[f"{fund}_BB_up"] = bb_up / price
    features[f"{fund}_BB_low"] = bb_low / price

    features[f"{fund}_ATR"] = atr / price

    features[f"{fund}_vol_change"] = volume[fund].pct_change()

features.fillna(method="ffill", inplace=True)
features.fillna(0, inplace=True)
features = features.clip(-10,10)
features = features.astype(np.float64)

# ==========================================
# Build Targets
# ==========================================

future_returns = returns.shift(-1)
features = features.iloc[:-1]

y_multi = []

for date in features.index:

    row = future_returns.loc[date]

    top = row.nlargest(top_n).index

    y_multi.append([1 if f in top else 0 for f in funds])

y_multi = np.array(y_multi)

# ==========================================
# Scaling
# ==========================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(features)

X = pd.DataFrame(X_scaled, index=features.index, columns=features.columns)

# ==========================================
# Build Sequences
# ==========================================

X_seq = []
y_seq = []

for i in range(lookback, len(X)):

    X_seq.append(X.iloc[i-lookback:i].values)

    y_seq.append(y_multi[i])

X_seq = np.array(X_seq)
y_seq = np.array(y_seq)

print("Total sequences:", len(X_seq))

# ==========================================
# Transformer Model
# ==========================================

def transformer_block(x, head_size=64, num_heads=4, ff_dim=128, dropout=0.3):

    attn = MultiHeadAttention(key_dim=head_size, num_heads=num_heads)(x,x)

    attn = Dropout(dropout)(attn)

    x = LayerNormalization(epsilon=1e-6)(x + attn)

    ff = Dense(ff_dim, activation="relu")(x)

    ff = Dense(x.shape[-1])(ff)

    x = LayerNormalization(epsilon=1e-6)(x + ff)

    return x

input_shape = (lookback, X.shape[1])

inputs = Input(shape=input_shape)

x = transformer_block(inputs)

x = transformer_block(x)

x = GlobalAveragePooling1D()(x)

x = Dense(128, activation="relu")(x)

x = Dropout(0.3)(x)

outputs = Dense(len(funds), activation="sigmoid")(x)

model = Model(inputs, outputs)

model.compile(optimizer=Adam(0.0005), loss="binary_crossentropy")

# EarlyStopping

early_stop = EarlyStopping(
    patience=3,
    restore_best_weights=True
)

# ==========================================
# RandomForest Model
# ==========================================

rf_base = RandomForestClassifier(
    n_estimators=400,
    max_depth=8,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42
)

rf = MultiOutputClassifier(rf_base)

# ==========================================
# Walk Forward Training
# ==========================================

predictions = []
test_indices = []

if len(X_seq) < train_window + test_window:

    print("Not enough sequences for walk-forward training.")

else:

    for start in range(train_window, len(X_seq)-test_window, test_window):

        X_train = X_seq[start-train_window:start]

        y_train = y_seq[start-train_window:start]

        X_test = X_seq[start:start+test_window]

        model.fit(
            X_train,
            y_train,
            epochs=10,
            batch_size=32,
            verbose=0,
            callbacks=[early_stop]
        )

        X_rf_train = X_train.reshape(X_train.shape[0], -1)

        X_rf_test = X_test.reshape(X_test.shape[0], -1)

        rf.fit(X_rf_train, y_train)

        transformer_probs = model.predict(X_test, verbose=0)

        rf_probs = rf.predict_proba(X_rf_test)

        rf_probs_array = np.hstack(
            [est[:, -1].reshape(-1,1) for est in rf_probs]
        )

        ensemble = (transformer_probs + rf_probs_array) / 2

        predictions.append(ensemble)

        test_indices.extend(range(start,start+test_window))

# ==========================================
# Stack Predictions
# ==========================================

predictions = np.vstack(predictions)

# ==========================================
# Backtesting
# ==========================================

test_index = features.index[lookback:][test_indices]

SPUS_200ma = close["SPUS"].rolling(200).mean()

strategy_returns = []
benchmark_returns = []

for i in range(0, len(predictions)-1, rebalance_period):

    next_date = test_index[i+1]

    if close.loc[next_date,"SPUS"] < SPUS_200ma.loc[next_date]:

        strategy_returns.append(0)

        benchmark_returns.append(
            returns.loc[next_date,"SPUS"]
        )

        continue

    probs = predictions[i]

    selected = probs.argsort()[-top_n:][::-1]

    vols = returns.loc[:next_date, funds].rolling(63).std().iloc[-1]

    inv_vol = 1 / vols[selected]

    weights = inv_vol / inv_vol.sum()

    allocation = np.zeros(len(funds))

    for idx,w in zip(selected,weights):

        allocation[idx] = w

    period_return = 0

    for j in range(rebalance_period):

        if i+j+1 >= len(test_index):
            break

        d = test_index[i+j+1]

        daily = sum(
            allocation[k] * returns.loc[d,funds[k]]
            for k in range(len(funds))
        )

        period_return += daily

    period_return -= transaction_cost

    strategy_returns.append(period_return)

    benchmark_returns.append(
        returns.loc[next_date,"SPUS"]
    )

strategy_cum = (1 + pd.Series(strategy_returns)).cumprod()

benchmark_cum = (1 + pd.Series(benchmark_returns)).cumprod()

# ==========================================
# Sharpe Ratio
# ==========================================

def sharpe_ratio(series):

    return np.sqrt(252) * series.mean() / series.std()

print("\nFinal Strategy Return:", round(strategy_cum.iloc[-1],4))

print("Final SPUS Return:", round(benchmark_cum.iloc[-1],4))

print("Strategy Sharpe:", round(sharpe_ratio(pd.Series(strategy_returns)),3))

print("SPUS Sharpe:", round(sharpe_ratio(pd.Series(benchmark_returns)),3))

# ==========================================
# Live Prediction
# ==========================================

latest_seq = X.iloc[-lookback:].values.reshape(
    1,lookback,X.shape[1]
)

transformer_today = model.predict(latest_seq)[0]

rf_today = rf.predict_proba(latest_seq.reshape(1,-1))

rf_today = np.array([est[0,-1] for est in rf_today])

ensemble_today = (transformer_today + rf_today) / 2

top_idx = ensemble_today.argsort()[-top_n:][::-1]

weights = ensemble_today[top_idx] / ensemble_today[top_idx].sum()

top_today = {
    funds[i]: round(float(w),3)
    for i,w in zip(top_idx,weights)
}

print("\nTop Picks Tomorrow:", top_today)

# ==========================================
# Save Models
# ==========================================

model.save("transformer_model.keras")

joblib.dump(rf,"rf_model.pkl")

joblib.dump(scaler,"scaler.pkl")

joblib.dump(X.columns.tolist(), "features.pkl")

