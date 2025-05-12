# Yahoo Finance API

A modern Python wrapper for Yahoo Finance with automatic dependency management.

## Features

- Automatic dependency management
- Real-time and historical market data
- Intraday, daily, weekly, and monthly data
- Company information and news sentiment
- Robust error handling
- Type hints and comprehensive documentation

## Installation

You can install the package directly from GitHub:

```bash
pip install git+https://github.com/MurselTasgin/alpha_seeker.git
```

## Quick Start for Yahoo Finance API Download

```python
from yahoo_finance_api import YahooFinanceAPI

# Initialize the API
api = YahooFinanceAPI()

# Get daily data for a symbol
daily_data = api.get_daily_data("AAPL")

# Get company overview
company_info = api.get_company_overview("AAPL")

# Get intraday data
intraday_data = api.get_intraday_data("AAPL", interval="5min")

# Get data for a specific date range
data = api.get_data_by_date_range(
    symbol="AAPL",
    start_date="2023-01-01",
    end_date="2023-12-31",
    interval="1d"
)
```

## API Reference

### YahooFinanceAPI

#### Methods

- `get_intraday_data(symbol: str, interval: str = "1min", outputsize: str = "compact") -> pd.DataFrame`
- `get_daily_data(symbol: str, outputsize: str = "compact") -> pd.DataFrame`
- `get_weekly_data(symbol: str) -> pd.DataFrame`
- `get_monthly_data(symbol: str) -> pd.DataFrame`
- `get_data_by_date_range(symbol: str, start_date: str, end_date: str, interval: str = "1d") -> pd.DataFrame`
- `get_company_overview(symbol: str) -> Dict[str, Any]`
- `get_news_sentiment(symbol: Optional[str] = None, topics: Optional[str] = None) -> Dict[str, Any]`
- `search_symbols(keywords: str) -> Dict[str, Any]`

For detailed documentation of each method, please see the docstrings in the code.

## Error Handling

The API includes several custom exception classes:

- `YahooFinanceError`: Base exception class
- `SymbolNotFoundError`: Raised when a symbol is not found
- `DataFetchError`: Raised when there's an error fetching data
- `EmptyDataError`: Raised when API returns empty data
- `InvalidInputError`: Raised for invalid inputs

## Development

To set up the development environment:

```bash
# Clone the repository
git clone https://github.com/MurselTasgin/alpha_seeker.git
cd alpha_seeker

# Install in development mode with development dependencies
pip install -e ".[dev]"

# Run tests
pytest
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.