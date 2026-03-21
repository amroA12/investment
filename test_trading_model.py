# test_trading_model.py
import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path


sys.path.append(str(Path(__file__).resolve().parent.parent))

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, Dense, Dropout, GlobalAveragePooling1D

# ==========================================
# Import functions and variables from the project
# ==========================================
from trading_model import (
    compute_RSI,
    compute_MACD,
    compute_ATR,
    bollinger,
    transformer_block,
    sharpe_ratio,
    lookback,
    X,
    y_multi,
    model,
    rf,
    funds
)

# ==========================================
# Fixtures for reusable data
# ==========================================
@pytest.fixture
def sample_series():
    return pd.Series(np.arange(1, 21))

@pytest.fixture
def sample_features():
    X_sample = np.random.rand(10, lookback, len(funds)*12)
    y_sample = np.random.randint(0,2,(10,len(funds)))
    return X_sample, y_sample

# ==========================================
# Technical indicators tests
# ==========================================
def test_compute_RSI(sample_series):
    rsi = compute_RSI(sample_series, period=5)
    assert isinstance(rsi, pd.Series)
    assert not rsi.isna().all()

def test_compute_MACD(sample_series):
    macd = compute_MACD(sample_series)
    assert isinstance(macd, pd.Series)
    assert not macd.isna().all()

def test_compute_ATR(sample_series):
    high = sample_series + 1
    low = sample_series - 1
    close = sample_series
    atr = compute_ATR(high, low, close, period=5)
    assert isinstance(atr, pd.Series)
    assert not atr.isna().all()

def test_bollinger(sample_series):
    upper, lower = bollinger(sample_series, window=5)
    assert isinstance(upper, pd.Series)
    assert isinstance(lower, pd.Series)
    # Ignore NaN values at the beginning
    valid = ~(upper.isna() | lower.isna())
    assert (upper[valid] >= lower[valid]).all()

# ==========================================
# Sharpe ratio test
# ==========================================
def test_sharpe_ratio():
    returns = pd.Series([0.01, 0.02, -0.005, 0.003])
    sr = sharpe_ratio(returns)
    assert isinstance(sr, float)

# ==========================================
# Transformer model output shape test
# ==========================================
def test_transformer_output_shape():
    input_shape = (lookback, X.shape[1])
    inputs = Input(shape=input_shape)
    x = transformer_block(inputs)
    x = transformer_block(x)
    x = GlobalAveragePooling1D()(x)  # Important: makes output 2D
    x = Dense(128, activation="relu")(x)
    x = Dense(len(funds), activation="sigmoid")(x)
    test_model = Model(inputs, x)
    test_model.compile(optimizer=Adam(0.001), loss="binary_crossentropy")

    sample_input = np.random.rand(1, lookback, X.shape[1])
    pred = test_model.predict(sample_input)
    assert pred.shape == (1, len(funds))
    assert (pred >= 0).all() and (pred <= 1).all()

# ==========================================
# RandomForest predict_proba shape test
# ==========================================
def test_rf_predict_proba_shape():
    X_rf = np.random.rand(5, lookback * X.shape[1])
    y_rf = np.random.randint(0,2,(5,len(funds)))

    rf_base = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
    rf_test = MultiOutputClassifier(rf_base)
    rf_test.fit(X_rf, y_rf)
    probs = rf_test.predict_proba(X_rf)
    assert len(probs) == len(funds)
    for est in probs:
        assert est.shape[0] == 5  # 5 samples
        assert est.shape[1] == 2  # probability for each class

# ==========================================
# Ensemble prediction test
# ==========================================
def test_ensemble_prediction():
    transformer_probs = np.random.rand(1, len(funds))
    rf_probs_array = np.random.rand(1, len(funds))
    ensemble = (transformer_probs + rf_probs_array) / 2
    assert ensemble.shape == (1, len(funds))
    assert (ensemble >= 0).all() and (ensemble <= 1).all()

# ==========================================
# Top picks allocation test
# ==========================================
def test_top_picks_allocation():
    ensemble_probs = np.random.rand(len(funds))
    top_n = 5
    top_idx = ensemble_probs.argsort()[-top_n:][::-1]
    weights = ensemble_probs[top_idx] / ensemble_probs[top_idx].sum()
    assert len(weights) == top_n
    assert np.isclose(weights.sum(), 1.0)
