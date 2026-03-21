from mock_data_provider import get_ticker_data
import pandas as pd
import joblib

from api import compute_RSI, scaler, model, rf, features_list

lookback = 42

close = get_ticker_data("AAPL")

df = pd.DataFrame()
df["return"] = close.pct_change()
df["ma10"] = close.rolling(10).mean()
df["ma20"] = close.rolling(20).mean()
df["rsi"] = compute_RSI(close)

df = df.fillna(0)

missing_cols = [col for col in features_list if col not in df.columns]

if missing_cols:
    missing_df = pd.DataFrame(
        0,
        index=df.index,
        columns=missing_cols
    )
    df = pd.concat([df, missing_df], axis=1)

df = df[features_list]

# إزالة fragmentation
df = df.copy()

df = df[features_list]

scaled = scaler.transform(df)

print("Scaled shape:", scaled.shape)

if len(scaled) >= lookback:
    seq = scaled[-lookback:].reshape(1, lookback, scaled.shape[1])

    t_pred = model.predict(seq)
    rf_pred = rf.predict_proba(seq.reshape(1, -1))

    print("Transformer OK")
    print("RF OK")
else:
    print("Not enough data")
