"""
Yahoo Finance API wrapper with automatic dependency management.
"""

from yahoo_finance_api.api import YahooFinanceAPI
from yahoo_finance_api.exceptions import (
    YahooFinanceError,
    SymbolNotFoundError,
    DataFetchError,
    EmptyDataError,
    InvalidInputError,
)

__version__ = "0.1.0"
__all__ = [
    "YahooFinanceAPI",
    "YahooFinanceError",
    "SymbolNotFoundError",
    "DataFetchError",
    "EmptyDataError",
    "InvalidInputError",
]