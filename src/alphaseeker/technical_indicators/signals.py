# src/technical_indicators/signals.py
"""
Implementation of trading signals based on technical indicators.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any
from .base import Signal, Indicator

class CrossoverSignal(Signal):
    """Signal generator for line crossovers."""
    
    def __init__(self, 
                 fast_indicator: Indicator,
                 slow_indicator: Indicator,
                 name: str = "Crossover"):
        """
        Initialize crossover signal generator.
        
        Args:
            fast_indicator: Faster-moving indicator
            slow_indicator: Slower-moving indicator
            name: Signal name
        """
        super().__init__(name)
        self.add_indicator(fast_indicator)
        self.add_indicator(slow_indicator)
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on crossovers.
        
        Returns:
            Series with signals (1 for buy, -1 for sell, 0 for hold)
        """
        fast = self._indicators[0].calculate(data)
        slow = self._indicators[1].calculate(data)
        
        signals = pd.Series(0, index=data.index)
        
        # Generate crossover signals
        signals[fast > slow] = 1
        signals[fast < slow] = -1
        
        # Only keep signal changes
        signals = signals.diff()
        signals = signals.replace(0, np.nan).fillna(0)
        
        return signals

class RSISignal(Signal):
    """Signal generator based on RSI levels."""
    
    def __init__(self,
                 rsi_indicator: Indicator,
                 overbought: float = 70,
                 oversold: float = 30,
                 name: str = "RSI"):
        """
        Initialize RSI signal generator.
        
        Args:
            rsi_indicator: RSI indicator instance
            overbought: Overbought level
            oversold: Oversold level
            name: Signal name
        """
        super().__init__(name)
        self.add_indicator(rsi_indicator)
        self.set_parameters(
            overbought=overbought,
            oversold=oversold
        )
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on RSI levels.
        
        Returns:
            Series with signals (1 for buy, -1 for sell, 0 for hold)
        """
        rsi = self._indicators[0].calculate(data)
        
        signals = pd.Series(0, index=data.index)
        
        # Generate signals based on RSI levels
        signals[rsi > self._parameters['overbought']] = -1
        signals[rsi < self._parameters['oversold']] = 1
        
        # Only keep signal changes
        signals = signals.diff()
        signals = signals.replace(0, np.nan).fillna(0)
        
        return signals

class MACDSignal(Signal):
    """Signal generator based on MACD."""
    
    def __init__(self,
                 macd_indicator: Indicator,
                 name: str = "MACD"):
        """
        Initialize MACD signal generator.
        
        Args:
            macd_indicator: MACD indicator instance
            name: Signal name
        """
        super().__init__(name)
        self.add_indicator(macd_indicator)
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on MACD crossovers.
        
        Returns:
            Series with signals (1 for buy, -1 for sell, 0 for hold)
        """
        macd_data = self._indicators[0].calculate(data)
        
        signals = pd.Series(0, index=data.index)
        
        # Generate signals based on MACD line crossing signal line
        signals[macd_data['MACD'] > macd_data['Signal']] = 1
        signals[macd_data['MACD'] < macd_data['Signal']] = -1
        
        # Only keep signal changes
        signals = signals.diff()
        signals = signals.replace(0, np.nan).fillna(0)
        
        return signals

class BollingerBandsSignal(Signal):
    """Signal generator based on Bollinger Bands."""
    
    def __init__(self,
                 bb_indicator: Indicator,
                 name: str = "BB"):
        """
        Initialize Bollinger Bands signal generator.
        
        Args:
            bb_indicator: Bollinger Bands indicator instance
            name: Signal name
        """
        super().__init__(name)
        self.add_indicator(bb_indicator)
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on price crossing Bollinger Bands.
        
        Returns:
            Series with signals (1 for buy, -1 for sell, 0 for hold)
        """
        bb_data = self._indicators[0].calculate(data)
        close = data['close']
        
        signals = pd.Series(0, index=data.index)
        
        # Generate signals based on price crossing bands
        signals[close < bb_data['Lower']] = 1  # Oversold
        signals[close > bb_data['Upper']] = -1  # Overbought
        
        # Only keep signal changes
        signals = signals.diff()
        signals = signals.replace(0, np.nan).fillna(0)
        
        return signals

class StochasticSignal(Signal):
    """Signal generator based on Stochastic Oscillator."""
    
    def __init__(self,
                 stoch_indicator: Indicator,
                 overbought: float = 80,
                 oversold: float = 20,
                 name: str = "Stochastic"):
        """
        Initialize Stochastic signal generator.
        
        Args:
            stoch_indicator: Stochastic indicator instance
            overbought: Overbought level
            oversold: Oversold level
            name: Signal name
        """
        super().__init__(name)
        self.add_indicator(stoch_indicator)
        self.set_parameters(
            overbought=overbought,
            oversold=oversold
        )
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on Stochastic crossovers.
        
        Returns:
            Series with signals (1 for buy, -1 for sell, 0 for hold)
        """
        stoch_data = self._indicators[0].calculate(data)
        
        signals = pd.Series(0, index=data.index)
        
        # Generate signals based on K crossing D
        signals[(stoch_data['K'] > stoch_data['D']) & 
               (stoch_data['K'] < self._parameters['oversold'])] = 1
        signals[(stoch_data['K'] < stoch_data['D']) & 
               (stoch_data['K'] > self._parameters['overbought'])] = -1
        
        # Only keep signal changes
        signals = signals.diff()
        signals = signals.replace(0, np.nan).fillna(0)
        
        return signals

class ADXSignal(Signal):
    """Signal generator based on ADX and DI lines."""
    
    def __init__(self,
                 adx_indicator: Indicator,
                 threshold: float = 25,
                 name: str = "ADX"):
        """
        Initialize ADX signal generator.
        
        Args:
            adx_indicator: ADX indicator instance
            threshold: ADX threshold for trend strength
            name: Signal name
        """
        super().__init__(name)
        self.add_indicator(adx_indicator)
        self.set_parameters(threshold=threshold)
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on ADX and DI crossovers.
        
        Returns:
            Series with signals (1 for buy, -1 for sell, 0 for hold)
        """
        adx_data = self._indicators[0].calculate(data)
        
        signals = pd.Series(0, index=data.index)
        
        # Generate signals when ADX indicates strong trend
        strong_trend = adx_data['ADX'] > self._parameters['threshold']
        
        # Buy when +DI crosses above -DI in strong trend
        signals[(adx_data['PDI'] > adx_data['NDI']) & 
               strong_trend] = 1
        
        # Sell when -DI crosses above +DI in strong trend
        signals[(adx_data['PDI'] < adx_data['NDI']) & 
               strong_trend] = -1
        
        # Only keep signal changes
        signals = signals.diff()
        signals = signals.replace(0, np.nan).fillna(0)
        
        return signals