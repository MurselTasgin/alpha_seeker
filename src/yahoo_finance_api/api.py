"""
Main API implementation for Yahoo Finance.
"""

import logging
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from functools import wraps

from yahoo_finance_api.package_manager import PackageManager
from yahoo_finance_api.exceptions import (
    YahooFinanceError,
    SymbolNotFoundError,
    DataFetchError,
    EmptyDataError,
    InvalidInputError,
)


# Set up logger
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("YahooFinanceAPI")

 

class YahooFinanceAPI:
    """A class for fetching financial data from Yahoo Finance."""
    
    # Required packages mapping (import name -> pip package name)
    REQUIRED_PACKAGES = {
        'yfinance': 'yfinance',
        'pandas': 'pandas',
        'curl_cffi': 'curl-cffi'
    }
    
    def __init__(self, session=None, timeout: int = 10):
        """
        Initialize the YahooFinanceAPI.
        
        Args:
            session: Optional session object for HTTP requests
            timeout: Request timeout in seconds
        """
        # First ensure all required packages are installed
        PackageManager.ensure_packages(self.REQUIRED_PACKAGES)

        # Now import the required packages
        try:
            import yfinance as yf
            import pandas as pd
            from curl_cffi import requests
            
            # Store the imported modules as instance variables
            self.yf = yf
            self.pd = pd
            self.requests = requests
        except ImportError as e:
            raise ImportError(f"Failed to import required packages after installation: {str(e)}")

        # Initialize session with impersonation if not provided
        self.session = self.requests.Session(impersonate="chrome")
        self.timeout = timeout
        self.logger = logger
        
    def _error_handler(func):
        """Decorator for consistent error handling across methods"""
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except ValueError as e:
                if "No data found" in str(e):
                    raise EmptyDataError(str(e))
                elif "Invalid" in str(e):
                    raise InvalidInputError(str(e))
                else:
                    raise DataFetchError(str(e))
            except Exception as e:
                # Log the exception
                self.logger.error(f"Error in {func.__name__}: {str(e)}")
                if "symbol" in str(e).lower() and "not found" in str(e).lower():
                    raise SymbolNotFoundError(f"Symbol not found: {args[0] if args else kwargs.get('symbol', '')}")
                elif "empty" in str(e).lower() or "no data" in str(e).lower():
                    raise EmptyDataError(f"No data available: {str(e)}")
                else:
                    raise DataFetchError(f"Error in {func.__name__}: {str(e)}")
        return wrapper
    
    def _validate_symbol(self, symbol: str) -> None:
        """
        Validate if a symbol is not empty and follows basic ticker format
        
        Args:
            symbol: Stock ticker symbol to validate
            
        Raises:
            InvalidInputError: If symbol is invalid
        """
        if not symbol:
            raise InvalidInputError("Symbol cannot be empty")
        
        if not isinstance(symbol, str):
            raise InvalidInputError(f"Symbol must be a string, got {type(symbol)}")
        
        # Basic validation - could be expanded
        if len(symbol) > 15:  # Most tickers are under 5, but some special ones can be longer
            raise InvalidInputError(f"Symbol '{symbol}' appears too long to be valid")
            
        # Check for invalid characters
        invalid_chars = set('!@#$%^&*()+={}[]|\\:;"<>,?/')
        if any(c in invalid_chars for c in symbol):
            raise InvalidInputError(f"Symbol '{symbol}' contains invalid characters")
    
    def _validate_date(self, date_str: str, param_name: str) -> None:
        """
        Validate a date string is in proper format and within reasonable bounds
        
        Args:
            date_str: Date string to validate
            param_name: Parameter name for error messages
            
        Raises:
            InvalidInputError: If date format is invalid or date is unreasonable
        """
        if not date_str:
            raise InvalidInputError(f"{param_name} cannot be empty")
            
        try:
            date = self.pd.to_datetime(date_str)
            
            # Check if date is too far in the past (before 1970)
            if date.year < 1970:
                raise InvalidInputError(f"{param_name} ({date_str}) is too far in the past. Must be after 1970.")
            
            # Check if date is in the future
            today = datetime.now()
            if date.date() > today.date():
                raise InvalidInputError(
                    f"{param_name} ({date_str}) is in the future. "
                    f"Please use a date on or before {today.strftime('%Y-%m-%d')}"
                )
                
        except Exception as e:
            if isinstance(e, InvalidInputError):
                raise e
            raise InvalidInputError(f"Invalid {param_name} format: '{date_str}'. Use 'YYYY-MM-DD' format.")
    
    def _validate_interval(self, interval: str, allowed_intervals: List[str]) -> None:
        """
        Validate interval is in allowed values
        
        Args:
            interval: Interval string to validate
            allowed_intervals: List of allowed interval values
            
        Raises:
            InvalidInputError: If interval is invalid
        """
        if not interval:
            raise InvalidInputError("Interval cannot be empty")
            
        if interval not in allowed_intervals:
            raise InvalidInputError(f"Invalid interval: '{interval}'. Allowed values: {', '.join(allowed_intervals)}")
    
    def _process_dataframe(self, df: 'pd.DataFrame', symbol: str) -> 'pd.DataFrame':
        """
        Process the dataframe to have a consistent format:
        - Reset index to make Date a column
        - Add symbol column
        - Flatten any multi-level column structure
        - Rename columns to lowercase
        
        Args:
            df: DataFrame to process
            symbol: Stock ticker symbol
            
        Returns:
            Processed DataFrame
        """
        # Check if dataframe is empty
        if df.empty:
            raise EmptyDataError(f"Empty dataframe received for symbol: {symbol}")
            
        # Reset index to make Date a regular column
        df = df.reset_index()
        
        # Add symbol column
        df['symbol'] = symbol
        
        # Handle multi-level columns if they exist
        if isinstance(df.columns, self.pd.MultiIndex):
            df.columns = [col[0].strip() for col in df.columns.values]
        
        # Rename columns to lowercase
        column_map = {
            'Date': 'date',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Adj Close': 'adjusted',
            'Volume': 'volume'
        }
        
        df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})
        
        # Ensure date is datetime
        df['date'] = self.pd.to_datetime(df['date'])
        
        # Reorder columns
        cols = ['date', 'symbol'] + [col for col in df.columns if col not in ['date', 'symbol']]
        df = df[cols]
        
        return df

    @_error_handler
    def get_intraday_data(self, symbol: str, interval: str = "1min", outputsize: str = "compact") -> 'pd.DataFrame':
        """
        Fetch intraday data using yfinance
        
        Args:
            symbol: Stock ticker symbol
            interval: Time interval for data ("1min", "5min", "15min", "30min", "60min")
            outputsize: Amount of data to fetch ("compact" or "full")
            
        Returns:
            DataFrame with intraday price data
            
        Raises:
            SymbolNotFoundError: If symbol doesn't exist
            EmptyDataError: If no data is found
            DataFetchError: For other errors during data fetch
        """
        # Validate inputs
        self._validate_symbol(symbol)
        
        # Convert AlphaVantage interval format to yfinance format
        interval_map = {"1min": "1m", "5min": "5m", "15min": "15m", "30min": "30m", "60min": "60m"}
        
        if interval not in interval_map:
            self._validate_interval(interval, list(interval_map.keys()))
            
        yf_interval = interval_map.get(interval, interval)
        
        # Map outputsize to period (compact = 7d, full = 60d for intraday)
        if outputsize not in ["compact", "full"]:
            raise InvalidInputError(f"Invalid outputsize: '{outputsize}'. Use 'compact' or 'full'.")
            
        period = "7d" if outputsize == "compact" else "60d"
        
        self.logger.info(f"Fetching intraday data for {symbol} with interval {yf_interval}")
        
        # Download data
        df = self.yf.download(
            symbol, 
            interval=yf_interval, 
            period=period, 
            session=self.session,
            progress=False
        )
        
        if df.empty:
            raise EmptyDataError(f"No intraday data found for {symbol} with interval {yf_interval}")
        
        # Process the dataframe to have a consistent format
        df = self._process_dataframe(df, symbol)
        
        return df

    @_error_handler
    def get_daily_data(self, symbol: str, outputsize: str = "compact") -> 'pd.DataFrame':
        """
        Fetch daily data using yfinance
        
        Args:
            symbol: Stock ticker symbol
            outputsize: Amount of data to fetch ("compact" or "full")
            
        Returns:
            DataFrame with daily price data
            
        Raises:
            SymbolNotFoundError: If symbol doesn't exist
            EmptyDataError: If no data is found
            DataFetchError: For other errors during data fetch
        """
        # Validate inputs
        self._validate_symbol(symbol)
        
        if outputsize not in ["compact", "full"]:
            raise InvalidInputError(f"Invalid outputsize: '{outputsize}'. Use 'compact' or 'full'.")
        
        # Map outputsize to period (compact = 100d, full = 5y for daily)
        period = "100d" if outputsize == "compact" else "5y"
        
        self.logger.info(f"Fetching daily data for {symbol}")
        
        # Download data
        df = self.yf.download(
            symbol, 
            interval="1d", 
            period=period, 
            session=self.session,
            progress=False
        )
        
        if df.empty:
            raise EmptyDataError(f"No daily data found for {symbol}")
        
        # Process the dataframe to have a consistent format
        df = self._process_dataframe(df, symbol)
        
        return df

    @_error_handler
    def get_weekly_data(self, symbol: str) -> 'pd.DataFrame':
        """
        Fetch weekly data using yfinance
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            DataFrame with weekly price data
            
        Raises:
            SymbolNotFoundError: If symbol doesn't exist
            EmptyDataError: If no data is found
            DataFetchError: For other errors during data fetch
        """
        # Validate inputs
        self._validate_symbol(symbol)
        
        self.logger.info(f"Fetching weekly data for {symbol}")
        
        # Download data
        df = self.yf.download(
            symbol, 
            interval="1wk", 
            period="5y", 
            session=self.session,
            progress=False
        )
        
        if df.empty:
            raise EmptyDataError(f"No weekly data found for {symbol}")
        
        # Process the dataframe to have a consistent format
        df = self._process_dataframe(df, symbol)
        
        return df

    @_error_handler
    def get_monthly_data(self, symbol: str) -> 'pd.DataFrame':
        """
        Fetch monthly data using yfinance
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            DataFrame with monthly price data
            
        Raises:
            SymbolNotFoundError: If symbol doesn't exist
            EmptyDataError: If no data is found
            DataFetchError: For other errors during data fetch
        """
        # Validate inputs
        self._validate_symbol(symbol)
        
        self.logger.info(f"Fetching monthly data for {symbol}")
        
        # Download data
        df = self.yf.download(
            symbol, 
            interval="1mo", 
            period="10y", 
            session=self.session,
            progress=False
        )
        
        if df.empty:
            raise EmptyDataError(f"No monthly data found for {symbol}")
        
        # Process the dataframe to have a consistent format
        df = self._process_dataframe(df, symbol)
        
        return df
            
    @_error_handler
    def get_data_by_date_range(self, symbol: str, start_date: str, end_date: str, interval: str = "1d") -> 'pd.DataFrame':
        """
        Fetch data for a specific date range
        
        Args:
            symbol: Stock ticker symbol
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            interval: Time interval for data ('1d', '1wk', '1mo', '1m', '5m', '15m', '30m', '60m')
                      Default is '1d' for daily data
                      
        Returns:
            DataFrame with date, symbol and price/volume data
            
        Raises:
            InvalidInputError: If inputs are invalid
            SymbolNotFoundError: If symbol doesn't exist
            EmptyDataError: If no data is found
            DataFetchError: For other errors during data fetch
        """
        # Validate inputs
        self._validate_symbol(symbol)
        self._validate_date(start_date, "start_date")
        self._validate_date(end_date, "end_date")
        
        allowed_intervals = ['1d', '1wk', '1mo', '1m', '5m', '15m', '30m', '60m']
        self._validate_interval(interval, allowed_intervals)
        
        # Convert dates to datetime objects for comparison
        start = self.pd.to_datetime(start_date)
        end = self.pd.to_datetime(end_date)
        
        # Check if start_date is after end_date
        if start > end:
            raise InvalidInputError(f"Start date ({start_date}) must be before end date ({end_date})")
        
        # Adjust end date to today if it's in the future
        today = datetime.now()
        if end.date() > today.date():
            self.logger.warning(
                f"End date ({end_date}) is in the future. "
                f"Adjusting to current date ({today.strftime('%Y-%m-%d')})"
            )
            end = today
            end_date = end.strftime('%Y-%m-%d')
        
        self.logger.info(f"Fetching {interval} data for {symbol} from {start_date} to {end_date}")
        
        try:
            # Download data
            df = self.yf.download(
                symbol, 
                start=start_date, 
                end=end_date, 
                interval=interval, 
                session=self.session,
                progress=False
            )
            
            if df.empty:
                # Try to determine if the symbol exists by fetching recent data
                test_df = self.yf.download(
                    symbol,
                    period="5d",  # Just get recent data to test
                    interval="1d",
                    session=self.session,
                    progress=False
                )
                
                if test_df.empty:
                    raise SymbolNotFoundError(f"Symbol '{symbol}' not found or may be delisted")
                else:
                    raise EmptyDataError(
                        f"No data available for {symbol} from {start_date} to {end_date} "
                        f"with interval {interval}. Try adjusting the date range or interval."
                    )
            
            # Process the dataframe to have a consistent format
            df = self._process_dataframe(df, symbol)
            
            return df
            
        except Exception as e:
            if isinstance(e, (SymbolNotFoundError, EmptyDataError)):
                raise e
            if "not found" in str(e).lower() or "delisted" in str(e).lower():
                raise SymbolNotFoundError(f"Symbol '{symbol}' not found or may be delisted")
            raise DataFetchError(f"Error fetching data: {str(e)}")

    @_error_handler
    def search_symbols(self, keywords: str) -> Dict[str, Any]:
        """
        Search for symbols matching the given keywords
        
        Note: yfinance doesn't provide a direct symbol search API. This is a limited implementation.
        For a comprehensive search, consider using yahoo_fin or a web scraping approach.
        
        Args:
            keywords: Search terms for finding symbols
            
        Returns:
            Dictionary with best matches for the keywords
            
        Raises:
            InvalidInputError: If keywords is empty
            DataFetchError: For errors during search
        """
        if not keywords:
            raise InvalidInputError("Keywords for search cannot be empty")
            
        self.logger.info(f"Searching for symbols matching: {keywords}")
        
        # Basic implementation - tries to get ticker info to see if it exists
        ticker = self.yf.Ticker(keywords, session=self.session)
        info = ticker.info
        
        # Validate response
        if not info or 'symbol' not in info:
            return {"bestMatches": []}
        
        # Found an exact match
        return {
            "bestMatches": [{
                "1. symbol": info.get('symbol', ''),
                "2. name": info.get('shortName', ''),
                "3. type": info.get('quoteType', ''),
                "4. region": info.get('region', ''),
                "5. marketOpen": info.get('regularMarketOpen', ''),
                "6. marketClose": info.get('regularMarketPreviousClose', ''),
                "7. timezone": info.get('exchangeTimezoneName', ''),
                "8. currency": info.get('currency', ''),
                "9. matchScore": "1.0000"
            }]
        }

    @_error_handler
    def get_company_overview(self, symbol: str) -> Dict[str, Any]:
        """
        Get company overview information using yfinance
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Dictionary with company information
            
        Raises:
            SymbolNotFoundError: If symbol doesn't exist
            EmptyDataError: If no company info is found
            DataFetchError: For other errors during data fetch
        """
        # Validate inputs
        self._validate_symbol(symbol)
        
        self.logger.info(f"Getting company overview for: {symbol}")
        
        # Get ticker info
        ticker = self.yf.Ticker(symbol, session=self.session)
        info = ticker.info
        
        if not info or 'symbol' not in info:
            raise EmptyDataError(f"No company information found for {symbol}")
        
        # Convert Yahoo Finance info to a format similar to AlphaVantage
        overview = {
            "Symbol": info.get('symbol', ''),
            "AssetType": info.get('quoteType', ''),
            "Name": info.get('shortName', ''),
            "Description": info.get('longBusinessSummary', ''),
            "Exchange": info.get('exchange', ''),
            "Currency": info.get('currency', ''),
            "Country": info.get('country', ''),
            "Sector": info.get('sector', ''),
            "Industry": info.get('industry', ''),
            "MarketCapitalization": info.get('marketCap', ''),
            "EBITDA": info.get('ebitda', ''),
            "PERatio": info.get('trailingPE', ''),
            "PEGRatio": info.get('pegRatio', ''),
            "BookValue": info.get('bookValue', ''),
            "DividendPerShare": info.get('dividendRate', ''),
            "DividendYield": info.get('dividendYield', ''),
            "EPS": info.get('trailingEps', ''),
            "RevenuePerShareTTM": info.get('revenuePerShare', ''),
            "ProfitMargin": info.get('profitMargins', ''),
            "52WeekHigh": info.get('fiftyTwoWeekHigh', ''),
            "52WeekLow": info.get('fiftyTwoWeekLow', ''),
            "50DayMovingAverage": info.get('fiftyDayAverage', ''),
            "200DayMovingAverage": info.get('twoHundredDayAverage', '')
        }
        
        return overview

    @_error_handler
    def get_news_sentiment(self, symbol: Optional[str] = None, topics: Optional[str] = None) -> Dict[str, Any]:
        """
        Get news data using yfinance
        
        Note: yfinance provides news but not sentiment analysis
        
        Args:
            symbol: Stock ticker symbol (optional)
            topics: News topics to search for (optional)
            
        Returns:
            Dictionary with news items
            
        Raises:
            InvalidInputError: If neither symbol nor topics is provided
            SymbolNotFoundError: If symbol doesn't exist
            DataFetchError: For other errors during data fetch
        """
        if not symbol and not topics:
            raise InvalidInputError("Either symbol or topics must be provided")
            
        self.logger.info(f"Getting news for: {symbol if symbol else topics}")
        
        if symbol:
            # Validate symbol if provided
            self._validate_symbol(symbol)
            
            # Get ticker news
            ticker = self.yf.Ticker(symbol, session=self.session)
            news_items = ticker.news
        else:
            # yfinance doesn't support topic-based news, return empty results
            return {"items": 0, "feed": []}
        
        if not news_items:
            return {"items": 0, "feed": []}
            
        feed = []
        for item in news_items:
            # Process each news item
            article = {
                "title": item.get('title', ''),
                "url": item.get('link', ''),
                "time_published": self._format_timestamp(item.get('providerPublishTime', '')),
                "authors": [item.get('publisher', '')],
                "summary": item.get('summary', ''),
                "source": item.get('publisher', ''),
                "category_within_source": "",
                "source_domain": item.get('publisher', ''),
                "topics": [],
                "overall_sentiment_score": None,  # Not provided by yfinance
                "overall_sentiment_label": "neutral"  # Default value
            }
            feed.append(article)
            
        return {
            "items": len(feed),
            "feed": feed
        }
    
    def _format_timestamp(self, timestamp: Union[int, str, None]) -> str:
        """
        Format a timestamp into ISO format
        
        Args:
            timestamp: Unix timestamp or datetime string
            
        Returns:
            Formatted datetime string
        """
        if not timestamp:
            return ""
            
        try:
            if isinstance(timestamp, (int, float)):
                dt = datetime.fromtimestamp(timestamp)
                return dt.isoformat()
            else:
                dt = self.pd.to_datetime(timestamp)
                return dt.isoformat()
        except Exception as e:
            self.logger.warning(f"Error formatting timestamp {timestamp}: {str(e)}")
            return str(timestamp)