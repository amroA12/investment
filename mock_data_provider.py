import pandas as pd
import numpy as np

# نفس الأسهم
FUNDS = [
    "AAPL","MSFT","GOOGL","AMD","ADBE",
    "CSCO","QCOM","TXN","INTU",
    "CRM","AVGO","ORCL","META","TSLA",
    "INTC","KLAC","ADI","SNPS","CDNS",
    "FTNT","SPUS","HLAL"
]

def generate_mock_data(days=100, seed=42, save_path="mock_data.csv"):
    np.random.seed(seed)

    dates = pd.date_range(end=pd.Timestamp.today(), periods=days)

    data = {}

    for stock in FUNDS:
        returns = np.random.normal(0.0005, 0.02, days)
        price = 100 * np.cumprod(1 + returns)

        high = price + np.random.uniform(0, 2, days)
        low = price - np.random.uniform(0, 2, days)
        volume = np.random.randint(100000, 1000000, days)

        data[("Close", stock)] = price
        data[("High", stock)] = high
        data[("Low", stock)] = low
        data[("Volume", stock)] = volume

    df = pd.DataFrame(data, index=dates)

    # تحويل الأعمدة إلى صيغة مناسبة للـ CSV
    df.columns = [f"{col[0]}_{col[1]}" for col in df.columns]

    # حفظ مباشرة في نفس المجلد الحالي
    df.to_csv(save_path)

    print(f"✅ Saved file in current directory: {save_path}")

    return df


if __name__ == "__main__":
    generate_mock_data()