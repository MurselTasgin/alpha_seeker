# src/technical_indicators/indicators.py
"""
Implementation of technical analysis indicators.
"""

from typing import Union
import pandas as pd
import numpy as np
from .base import Indicator

class MovingAverage(Indicator):
    """Simple Moving Average (SMA) indicator."""
    
    def __init__(self, period: int = 20, column: str = 'close'):
        """
        Initialize SMA indicator.
        
        Args:
            period: Moving average period
            column: Column to calculate MA for
        """
        super().__init__(f"SMA_{period}")
        self.set_parameters(period=period, column=column)
    
    def calculate(self, data: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        """Calculate SMA values."""
        if isinstance(data, pd.Series):
            series = data
        else:
            series = data[self._parameters['column']]
            
        self._series = series.rolling(
            window=self._parameters['period'],
            min_periods=1
        ).mean()
        return self._series

class ExponentialMovingAverage(Indicator):
    """Exponential Moving Average (EMA) indicator."""
    
    def __init__(self, period: int = 20, column: str = 'close'):
        """
        Initialize EMA indicator.
        
        Args:
            period: Moving average period
            column: Column to calculate EMA for
        """
        super().__init__(f"EMA_{period}")
        self.set_parameters(period=period, column=column)
    
    def calculate(self, data: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        """Calculate EMA values."""
        if isinstance(data, pd.Series):
            series = data
        else:
            series = data[self._parameters['column']]
            
        self._series = series.ewm(
            span=self._parameters['period'],
            adjust=False
        ).mean()
        return self._series

class RSI(Indicator):
    """Relative Strength Index (RSI) indicator."""
    
    def __init__(self, period: int = 14, column: str = 'close'):
        """
        Initialize RSI indicator.
        
        Args:
            period: RSI period
            column: Column to calculate RSI for
        """
        super().__init__(f"RSI_{period}")
        self.set_parameters(period=period, column=column)
    
    def calculate(self, data: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        """Calculate RSI values."""
        if isinstance(data, pd.Series):
            series = data
        else:
            series = data[self._parameters['column']]
            
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(
            window=self._parameters['period']
        ).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(
            window=self._parameters['period']
        ).mean()
        
        rs = gain / loss
        self._series = 100 - (100 / (1 + rs))
        return self._series

class MACD(Indicator):
    """Moving Average Convergence Divergence (MACD) indicator."""
    
    def __init__(self, 
                 fast_period: int = 12,
                 slow_period: int = 26,
                 signal_period: int = 9,
                 column: str = 'close'):
        """
        Initialize MACD indicator.
        
        Args:
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line period
            column: Column to calculate MACD for
        """
        super().__init__("MACD")
        self.set_parameters(
            fast_period=fast_period,
            slow_period=slow_period,
            signal_period=signal_period,
            column=column
        )
        self._dataframe = None
    
    def calculate(self, data: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
        """
        Calculate MACD values.
        
        Returns DataFrame with columns:
        - MACD: MACD line
        - Signal: Signal line
        - Histogram: MACD histogram
        """
        if isinstance(data, pd.Series):
            series = data
        else:
            series = data[self._parameters['column']]
            
        fast_ema = series.ewm(
            span=self._parameters['fast_period'],
            adjust=False
        ).mean()
        slow_ema = series.ewm(
            span=self._parameters['slow_period'],
            adjust=False
        ).mean()
        
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(
            span=self._parameters['signal_period'],
            adjust=False
        ).mean()
        histogram = macd_line - signal_line
        
        self._dataframe = pd.DataFrame({
            'MACD': macd_line,
            'Signal': signal_line,
            'Histogram': histogram
        })
        self._series = macd_line
        
        return self._dataframe

class BollingerBands(Indicator):
    """Bollinger Bands indicator."""
    
    def __init__(self,
                 period: int = 20,
                 std_dev: float = 2.0,
                 column: str = 'close'):
        """
        Initialize Bollinger Bands indicator.
        
        Args:
            period: Moving average period
            std_dev: Number of standard deviations
            column: Column to calculate bands for
        """
        super().__init__("BB")
        self.set_parameters(
            period=period,
            std_dev=std_dev,
            column=column
        )
        self._dataframe = None
    
    def calculate(self, data: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
        """
        Calculate Bollinger Bands values.
        
        Returns DataFrame with columns:
        - Middle: Middle band (SMA)
        - Upper: Upper band
        - Lower: Lower band
        """
        if isinstance(data, pd.Series):
            series = data
        else:
            series = data[self._parameters['column']]
            
        middle = series.rolling(
            window=self._parameters['period']
        ).mean()
        std = series.rolling(
            window=self._parameters['period']
        ).std()
        
        upper = middle + (std * self._parameters['std_dev'])
        lower = middle - (std * self._parameters['std_dev'])
        
        self._dataframe = pd.DataFrame({
            'Middle': middle,
            'Upper': upper,
            'Lower': lower
        })
        self._series = middle
        
        return self._dataframe

class Stochastic(Indicator):
    """Stochastic Oscillator indicator."""
    
    def __init__(self,
                 k_period: int = 14,
                 d_period: int = 3,
                 slow: bool = True):
        """
        Initialize Stochastic Oscillator.
        
        Args:
            k_period: %K period
            d_period: %D period
            slow: Whether to calculate slow stochastic
        """
        super().__init__("STOCH")
        self.set_parameters(
            k_period=k_period,
            d_period=d_period,
            slow=slow
        )
        self._dataframe = None
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Stochastic Oscillator values.
        
        Returns DataFrame with columns:
        - K: %K line
        - D: %D line
        """
        high_period = data['high'].rolling(
            window=self._parameters['k_period']
        ).max()
        low_period = data['low'].rolling(
            window=self._parameters['k_period']
        ).min()
        
        k = 100 * (data['close'] - low_period) / (high_period - low_period)
        
        if self._parameters['slow']:
            k = k.rolling(window=3).mean()
            
        d = k.rolling(window=self._parameters['d_period']).mean()
        
        self._dataframe = pd.DataFrame({
            'K': k,
            'D': d
        })
        self._series = k
        
        return self._dataframe

class ADX(Indicator):
    """Average Directional Index (ADX) indicator."""
    
    def __init__(self, period: int = 14):
        """
        Initialize ADX indicator.
        
        Args:
            period: ADX period
        """
        super().__init__(f"ADX_{period}")
        self.set_parameters(period=period)
        self._dataframe = None
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate ADX values.
        
        Returns DataFrame with columns:
        - ADX: Average Directional Index
        - PDI: Positive Directional Indicator
        - NDI: Negative Directional Indicator
        """
        high = data['high']
        low = data['low']
        close = data['close']
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.DataFrame({'TR1': tr1, 'TR2': tr2, 'TR3': tr3}).max(axis=1)
        
        # Directional Movement
        up = high - high.shift(1)
        down = low.shift(1) - low
        
        pos_dm = np.where((up > down) & (up > 0), up, 0)
        neg_dm = np.where((down > up) & (down > 0), down, 0)
        
        # Smoothed TR and DM
        period = self._parameters['period']
        smoothed_tr = tr.rolling(window=period).sum()
        smoothed_pos_dm = pd.Series(pos_dm).rolling(window=period).sum()
        smoothed_neg_dm = pd.Series(neg_dm).rolling(window=period).sum()
        
        # Directional Indicators
        pdi = 100 * smoothed_pos_dm / smoothed_tr
        ndi = 100 * smoothed_neg_dm / smoothed_tr
        
        # ADX
        dx = 100 * abs(pdi - ndi) / (pdi + ndi)
        adx = dx.rolling(window=period).mean()
        
        self._dataframe = pd.DataFrame({
            'ADX': adx,
            'PDI': pdi,
            'NDI': ndi
        })
        self._series = adx
        
        return self._dataframe