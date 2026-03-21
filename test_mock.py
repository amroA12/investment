from mock_data_provider import get_ticker_data

data = get_ticker_data("AAPL")

print(data.head())
print("\nLength:", len(data))