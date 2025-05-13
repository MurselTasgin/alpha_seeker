# src/technical_indicators/base.py
"""
Base classes for technical analysis indicators and signals.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
import pandas as pd
import numpy as np

class Indicator(ABC):
    """Abstract base class for technical indicators."""
    
    def __init__(self, name: str):
        """
        Initialize the indicator.
        
        Args:
            name: Unique identifier for the indicator
        """
        self.name = name
        self._parameters: Dict[str, Any] = {}
        self._series: Optional[pd.Series] = None
        self._dataframe: Optional[pd.DataFrame] = None
    
    @abstractmethod
    def calculate(self, data: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        """
        Calculate the indicator values.
        
        Args:
            data: Price/volume data
            
        Returns:
            Series with indicator values
        """
        pass
    
    def set_parameters(self, **kwargs) -> 'Indicator':
        """
        Set indicator parameters.
        
        Args:
            **kwargs: Parameter name-value pairs
            
        Returns:
            Self for method chaining
        """
        self._parameters.update(kwargs)
        return self
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get current parameter values."""
        return self._parameters.copy()
    
    def validate_parameters(self) -> bool:
        """
        Validate parameter values.
        
        Returns:
            True if parameters are valid
        """
        return True
    
    def get_value(self, index: int = -1) -> float:
        """
        Get indicator value at specific index.
        
        Args:
            index: Position in the series (-1 for latest)
            
        Returns:
            Indicator value
        """
        if self._series is None:
            raise ValueError("Indicator hasn't been calculated yet")
        return self._series.iloc[index]
    
    def get_series(self) -> pd.Series:
        """Get full indicator series."""
        if self._series is None:
            raise ValueError("Indicator hasn't been calculated yet")
        return self._series.copy()

class Signal(ABC):
    """Abstract base class for trading signals."""
    
    def __init__(self, name: str):
        """
        Initialize the signal generator.
        
        Args:
            name: Unique identifier for the signal
        """
        self.name = name
        self._parameters: Dict[str, Any] = {}
        self._indicators: List[Indicator] = []
    
    @abstractmethod
    def generate(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals.
        
        Args:
            data: Price/volume data with indicator values
            
        Returns:
            Series with signal values (1 for buy, -1 for sell, 0 for hold)
        """
        pass
    
    def add_indicator(self, indicator: Indicator) -> 'Signal':
        """
        Add an indicator to the signal generator.
        
        Args:
            indicator: Technical indicator instance
            
        Returns:
            Self for method chaining
        """
        self._indicators.append(indicator)
        return self
    
    def set_parameters(self, **kwargs) -> 'Signal':
        """
        Set signal parameters.
        
        Args:
            **kwargs: Parameter name-value pairs
            
        Returns:
            Self for method chaining
        """
        self._parameters.update(kwargs)
        return self
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get current parameter values."""
        return self._parameters.copy()

class Strategy(ABC):
    """Abstract base class for trading strategies."""
    
    def __init__(self, name: str):
        """
        Initialize the strategy.
        
        Args:
            name: Unique identifier for the strategy
        """
        self.name = name
        self._signals: List[Signal] = []
        self._parameters: Dict[str, Any] = {}
    
    @abstractmethod
    def evaluate(self, data: pd.DataFrame) -> pd.Series:
        """
        Evaluate the strategy and generate trading decisions.
        
        Args:
            data: Price/volume data with indicator values
            
        Returns:
            Series with strategy decisions (1 for buy, -1 for sell, 0 for hold)
        """
        pass
    
    def add_signal(self, signal: Signal) -> 'Strategy':
        """
        Add a signal to the strategy.
        
        Args:
            signal: Signal generator instance
            
        Returns:
            Self for method chaining
        """
        self._signals.append(signal)
        return self
    
    def set_parameters(self, **kwargs) -> 'Strategy':
        """
        Set strategy parameters.
        
        Args:
            **kwargs: Parameter name-value pairs
            
        Returns:
            Self for method chaining
        """
        self._parameters.update(kwargs)
        return self
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get current parameter values."""
        return self._parameters.copy()