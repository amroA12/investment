import pandas as pd
from predict import predict_portfolio

# ==========================================
# 1. Load sample data (user input)
# ==========================================
data = pd.read_csv("mock_data.csv", index_col=0)

data = data.select_dtypes(include=['number'])

print("Input Data (Cleaned):")
print(data.head())

# ==========================================
# 2. Run Portfolio Prediction
# ==========================================
portfolio = predict_portfolio(data)

# ==========================================
# 3. Output Results
# ==========================================
print("\nRecommended Portfolio Allocation:")

if isinstance(portfolio, dict):
    for asset, weight in portfolio.items():
        print(f"{asset}: {round(weight * 100, 2)}%")
else:
    print(portfolio)
