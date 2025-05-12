"""
Custom exceptions for the Yahoo Finance API.
"""

class YahooFinanceError(Exception):
    """Base exception class for YahooFinanceAPI errors"""
    pass

class SymbolNotFoundError(YahooFinanceError):
    """Exception raised when a symbol is not found"""
    pass

class DataFetchError(YahooFinanceError):
    """Exception raised when there's an error fetching data"""
    pass

class EmptyDataError(YahooFinanceError):
    """Exception raised when API returns empty data"""
    pass

class InvalidInputError(YahooFinanceError):
    """Exception raised for invalid inputs"""
    pass