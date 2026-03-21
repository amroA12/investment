import pandas as pd
from predict import predict_portfolio


def test_user_flow():
    data = pd.read_csv("sample_data.csv")

    result = predict_portfolio(data)

    assert isinstance(result, dict)
    assert len(result) == 5

    total_weight = sum(result.values())
    assert abs(total_weight - 1.0) < 1e-6
