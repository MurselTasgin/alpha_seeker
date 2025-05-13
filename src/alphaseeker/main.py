from data_providers.yahoo_finance_api import YahooFinanceAPI

my_dataAPI = YahooFinanceAPI()

symbol_name = 'QQQI'
start_date = '2025-01-01'
end_date = '2025-05-08'

ticker_df = my_dataAPI.get_data_by_date_range(symbol_name, start_date=start_date, end_date=end_date) 

print(ticker_df.head())