📊 Stock Portfolio Prediction API
📌 Overview

This project is an end-to-end machine learning system for stock portfolio prediction and allocation using:

Transformer Neural Network
Random Forest (Multi-output)
Custom Mock Market Data (replacing external APIs like yfinance)

The system generates synthetic stock data, preprocesses it, and predicts optimal portfolio weights.

🚀 Features
✅ Portfolio prediction using ensemble learning
✅ Fully offline mock market data (no external dependencies)
✅ FastAPI REST API for predictions
✅ Data generation similar to real market behavior
✅ Scalable and modular architecture
✅ Unit tests included
✅ Compatible with trained models (Transformer + Random Forest)
📁 Project Structure
project/
│
├── api.py
├── predict.py
├── trading_model.py
├── mock_data_provider.py   
├── mock_data.csv           
│
├── example_usage.py
├── sample_data.csv
│
├── test_model_usage.py
├── test_trading_model.py
│
├── transformer_model.keras
├── rf_model.pkl
├── scaler.pkl
├── features.pkl
│
└── README.md
⚙️ Installation
pip install -r requirements.txt
▶️ Run API
uvicorn api:app --reload

Then open:

http://127.0.0.1:8000/docs
📊 Data Source (Important Update)

This project DOES NOT use yfinance anymore.

Instead, it uses a custom-built generator:

mock_data_provider.py
Generates realistic stock-like data
Saves data to mock_data.csv
Mimics:
Price movement
Volume
High / Low / Close
📥 Generate Data
python mock_data_provider.py

Output:

mock_data.csv
📈 Data Format

Each column represents a feature:

Close_AAPL
High_AAPL
Low_AAPL
Volume_AAPL
Close_MSFT
...

Each row represents one timestep (daily data).

🧠 How It Works
Generate mock stock data
Load and preprocess data
Select features
Scale data using saved scaler
Feed into models:
Transformer
Random Forest
Combine predictions (Ensemble)
Return top asset allocations
📥 API Usage
Endpoint
POST /predict
Input Format
{
  "ticker": "AAPL"
}

or (depending on your implementation):

{
  "data": [...]
}
Example Response
{
  "top_picks": {
    "AAPL": 0.25,
    "MSFT": 0.20,
    "TSLA": 0.15
  }
}
👤 Example Usage
python example_usage.py
🧪 Running Tests
pytest
🧾 Model Files

The project uses pre-trained models:

File	Description
transformer_model.keras	Deep learning model
rf_model.pkl	Random Forest model
scaler.pkl	Feature scaler
features.pkl	Feature order (important)
⚠️ Important Notes
1. Feature Matching Problem (Fixed Concept)
Features used in prediction must match training features
Order of columns must be identical
2. Data Window

Minimum required input:

42 timesteps
This is the lookback window used in the model
3. Mock vs Real Data
Type	Description
Mock Data	Generated locally for testing
Real Data	Requires APIs like yfinance
🔧 Key Components
1. Mock Data Provider

Responsible for generating synthetic stock data:

No internet required
Mimics real market behavior
Saves CSV for reuse
2. Trading Model
Handles feature engineering
Combines ML models
Outputs portfolio allocation
3. FastAPI Layer
Exposes prediction endpoint
Handles requests and responses
Loads models once (optimized)
🧠 Workflow
Generate Data → Preprocess → Scale → Predict → Ensemble → Output
📬 Contact

For improvements or contributions, feel free to extend the project.

🔥 Future Improvements
Integration with real market APIs
Advanced indicators (RSI, MACD)
Reinforcement learning portfolio optimization
Real-time predictions
Model retraining pipeline
