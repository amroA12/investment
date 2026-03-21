from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

from mock_data_provider import get_ticker_data

# =========================
# Load Models (مرة واحدة فقط)
# =========================

model = tf.keras.models.load_model("transformer_model.keras")
rf = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")
features_list = joblib.load("features.pkl")

lookback = 42

# نفس ترتيب الأسهم
funds = [
    "AAPL","MSFT","GOOGL","AMD","ADBE",
    "CSCO","QCOM","TXN","INTU",
    "CRM","AVGO","ORCL","META","TSLA",
    "INTC","KLAC","ADI","SNPS","CDNS",
    "FTNT","SPUS","HLAL"
]

# =========================
# App
# =========================

app = FastAPI()

# =========================
# Request Model
# =========================

class TickerRequest(BaseModel):
    ticker: str

# =========================
# Indicators
# =========================

def compute_RSI(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss
    return 100 - (100/(1+rs))

# =========================
# Prediction Endpoint
# =========================

@app.post("/predict")
def predict(req: TickerRequest):

    try:
        ticker = req.ticker.upper()

        # =====================
        # بيانات بديلة
        # =====================

        close = get_ticker_data(ticker)

        if close.empty:
            return {"error": "No data found"}

        # =====================
        # Feature Engineering
        # =====================

        df = pd.DataFrame()

        df["return"] = close.pct_change()
        df["ma10"] = close.rolling(10).mean()
        df["ma20"] = close.rolling(20).mean()
        df["rsi"] = compute_RSI(close)

        df = df.fillna(0)

        # =====================
        # حل fragmentation + mismatch features
        # =====================

        missing_cols = [col for col in features_list if col not in df.columns]

        if missing_cols:
            missing_df = pd.DataFrame(
                0,
                index=df.index,
                columns=missing_cols
            )
            df = pd.concat([df, missing_df], axis=1)

        df = df[features_list]
        df = df.copy()  # إزالة fragmentation

        # =====================
        # Scaling
        # =====================

        scaled = scaler.transform(df)

        # =====================
        # Sequence
        # =====================

        if len(scaled) < lookback:
            return {"error": "Not enough data"}

        seq = scaled[-lookback:]
        seq = seq.reshape(1, lookback, scaled.shape[1])

        # =====================
        # Prediction
        # =====================

        transformer_pred = model.predict(seq, verbose=0)[0]

        rf_input = seq.reshape(1, -1)
        rf_probs = rf.predict_proba(rf_input)
        rf_pred = np.array([est[0, -1] for est in rf_probs])

        ensemble = (transformer_pred + rf_pred) / 2

        # =====================
        # Top Picks
        # =====================

        top_n = 5
        top_idx = ensemble.argsort()[-top_n:][::-1]

        weights = ensemble[top_idx] / ensemble[top_idx].sum()

        top_assets = {
            funds[i]: float(w)
            for i, w in zip(top_idx, weights)
        }

        return {
            "ticker": ticker,
            "top_picks": top_assets,
            "prediction": ensemble.tolist()
        }

    except Exception as e:
        return {"error": str(e)}
