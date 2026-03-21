import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

lookback = 42
top_n = 5

# =========================
# Load models ONCE
# =========================

transformer = tf.keras.models.load_model("transformer_model.keras")
rf = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")


def build_features_from_prices(df):

    out = pd.DataFrame()

    for col in df.columns:

        price = df[col]

        out[f"{col}_ret_1w"] = price.pct_change(5)
        out[f"{col}_ret_1m"] = price.pct_change(21)
        out[f"{col}_ret_3m"] = price.pct_change(63)

        out[f"{col}_vol_1m"] = price.pct_change().rolling(21).std()
        out[f"{col}_vol_3m"] = price.pct_change().rolling(63).std()

        ma10 = price.rolling(10).mean()
        ma50 = price.rolling(50).mean()
        ma200 = price.rolling(200).mean()

        out[f"{col}_trend_10_50"] = ma10 - ma50
        out[f"{col}_trend_50_200"] = ma50 - ma200

        delta = price.diff()

        gain = delta.clip(lower=0).rolling(14).mean()
        loss = -delta.clip(upper=0).rolling(14).mean()

        rs = gain / loss

        out[f"{col}_RSI"] = 100 - (100 / (1 + rs))

        out[f"{col}_MACD"] = price.ewm(span=12).mean() - price.ewm(span=26).mean()

        ma20 = price.rolling(20).mean()
        std20 = price.rolling(20).std()

        out[f"{col}_BB_up"] = ma20 + 2 * std20
        out[f"{col}_BB_low"] = ma20 - 2 * std20

        out[f"{col}_ATR"] = price.rolling(14).std()

        out[f"{col}_vol_change"] = price.pct_change().rolling(5).std()

    return out.fillna(0)


def predict_portfolio(data):

    if not set(features).issubset(set(data.columns)):
        data = build_features_from_prices(data)

    for col in features:
        if col not in data.columns:
            data[col] = 0

    data = data[features]

    X = scaler.transform(data)

    if len(X) < lookback:
        padding = np.zeros((lookback - len(X), X.shape[1]))
        seq_data = np.vstack([padding, X])
    else:
        seq_data = X[-lookback:]

    seq = seq_data.reshape(1, lookback, X.shape[1])

    transformer_pred = transformer.predict(seq)[0]

    rf_pred = rf.predict_proba(seq.reshape(1, -1))
    rf_pred = np.array([est[0, -1] for est in rf_pred])

    ensemble = (transformer_pred + rf_pred) / 2

    top_idx = ensemble.argsort()[-top_n:][::-1]

    weights = ensemble[top_idx] / ensemble[top_idx].sum()

    return {
        f"asset_{i}": float(w)
        for i, w in zip(top_idx, weights)
    }
