# src/technical_indicators/strategies.py
"""
Implementation of trading strategies combining multiple signals.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any
from .base import Strategy, Signal

class CompositeStrategy(Strategy):
    """Strategy that combines multiple signals with weights."""
    
    def __init__(self,
                 name: str,
                 signal_weights: Dict[str, float] = None):
        """
        Initialize composite strategy.
        
        Args:
            name: Strategy name
            signal_weights: Dictionary mapping signal names to weights
        """
        super().__init__(name)
        self.set_parameters(
            weights=signal_weights or {},
            threshold=0.5
        )
    
    def evaluate(self, data: pd.DataFrame) -> pd.Series:
        """
        Evaluate strategy by combining weighted signals.
        
        Returns:
            Series with strategy decisions (1 for buy, -1 for sell, 0 for hold)
        """
        if not self._signals:
            raise ValueError("No signals added to strategy")
        
        # Generate all signals
        signals = pd.DataFrame()
        weights = []
        
        for signal in self._signals:
            sig = signal.generate(data)
            signals[signal.name] = sig
            weights.append(
                self._parameters['weights'].get(signal.name, 1.0)
            )
            
        # Normalize weights
        weights = np.array(weights) / sum(weights)
        
        # Combine signals
        combined = signals.multiply(weights, axis=1).sum(axis=1)
        
        # Apply threshold
        threshold = self._parameters['threshold']
        decisions = pd.Series(0, index=data.index)
        decisions[combined > threshold] = 1
        decisions[combined < -threshold] = -1
        
        return decisions

class TrendFollowingStrategy(Strategy):
    """Strategy that follows established trends."""
    
    def __init__(self,
                 name: str,
                 confirmation_period: int = 2):
        """
        Initialize trend following strategy.
        
        Args:
            name: Strategy name
            confirmation_period: Number of periods to confirm trend
        """
        super().__init__(name)
        self.set_parameters(confirmation_period=confirmation_period)
    
    def evaluate(self, data: pd.DataFrame) -> pd.Series:
        """
        Evaluate strategy by confirming trend signals.
        
        Returns:
            Series with strategy decisions (1 for buy, -1 for sell, 0 for hold)
        """
        if not self._signals:
            raise ValueError("No signals added to strategy")
        
        # Generate all signals
        signals = pd.DataFrame()
        
        for signal in self._signals:
            signals[signal.name] = signal.generate(data)
        
        # Count consistent signals
        bull_count = (signals == 1).sum(axis=1)
        bear_count = (signals == -1).sum(axis=1)
        
        # Make decisions based on signal consistency
        decisions = pd.Series(0, index=data.index)
        min_signals = self._parameters['confirmation_period']
        
        decisions[bull_count >= min_signals] = 1
        decisions[bear_count >= min_signals] = -1
        
        return decisions

class MeanReversionStrategy(Strategy):
    """Strategy that trades mean reversion patterns."""
    
    def __init__(self,
                 name: str,
                 oversold_threshold: float = -0.5,
                 overbought_threshold: float = 0.5):
        """
        Initialize mean reversion strategy.
        
        Args:
            name: Strategy name
            oversold_threshold: Threshold for oversold condition
            overbought_threshold: Threshold for overbought condition
        """
        super().__init__(name)
        self.set_parameters(
            oversold=oversold_threshold,
            overbought=overbought_threshold
        )
    
    def evaluate(self, data: pd.DataFrame) -> pd.Series:
        """
        Evaluate strategy by identifying mean reversion opportunities.
        
        Returns:
            Series with strategy decisions (1 for buy, -1 for sell, 0 for hold)
        """
        if not self._signals:
            raise ValueError("No signals added to strategy")
        
        # Generate all signals
        signals = pd.DataFrame()
        
        for signal in self._signals:
            signals[signal.name] = signal.generate(data)
        
        # Calculate average signal
        avg_signal = signals.mean(axis=1)
        
        # Make decisions based on extreme readings
        decisions = pd.Series(0, index=data.index)
        
        decisions[avg_signal <= self._parameters['oversold']] = 1
        decisions[avg_signal >= self._parameters['overbought']] = -1
        
        return decisions

class BreakoutStrategy(Strategy):
    """Strategy that trades breakouts with volume confirmation."""
    
    def __init__(self,
                 name: str,
                 volume_factor: float = 1.5):
        """
        Initialize breakout strategy.
        
        Args:
            name: Strategy name
            volume_factor: Volume increase factor for confirmation
        """
        super().__init__(name)
        self.set_parameters(
            volume_factor=volume_factor,
            lookback_period=20
        )
    
    def evaluate(self, data: pd.DataFrame) -> pd.Series:
        """
        Evaluate strategy by identifying breakouts with volume confirmation.
        
        Returns:
            Series with strategy decisions (1 for buy, -1 for sell, 0 for hold)
        """
        if not self._signals:
            raise ValueError("No signals added to strategy")
        
        # Generate all signals
        signals = pd.DataFrame()
        for signal in self._signals:
            signals[signal.name] = signal.generate(data)
        
        # Calculate volume conditions
        avg_volume = data['volume'].rolling(
            window=self._parameters['lookback_period']
        ).mean()
        volume_breakout = data['volume'] > (
            avg_volume * self._parameters['volume_factor']
        )
        
        # Combine signals with volume confirmation
        decisions = pd.Series(0, index=data.index)
        
        # Buy on upside breakout with volume confirmation
        decisions[(signals.mean(axis=1) > 0) & volume_breakout] = 1
        
        # Sell on downside breakout with volume confirmation
        decisions[(signals.mean(axis=1) < 0) & volume_breakout] = -1
        
        return decisions

class DivergenceStrategy(Strategy):
    """Strategy that trades price-indicator divergences."""
    
    def __init__(self,
                 name: str,
                 divergence_threshold: float = 0.1,
                 confirmation_periods: int = 3):
        """
        Initialize divergence strategy.
        
        Args:
            name: Strategy name
            divergence_threshold: Minimum divergence to trigger signal
            confirmation_periods: Number of periods to confirm divergence
        """
        super().__init__(name)
        self.set_parameters(
            threshold=divergence_threshold,
            confirmation=confirmation_periods
        )
    
    def evaluate(self, data: pd.DataFrame) -> pd.Series:
        """
        Evaluate strategy by identifying price-indicator divergences.
        
        Returns:
            Series with strategy decisions (1 for buy, -1 for sell, 0 for hold)
        """
        if not self._signals:
            raise ValueError("No signals added to strategy")
        
        decisions = pd.Series(0, index=data.index)
        price_changes = data['close'].pct_change()
        
        for signal in self._signals:
            indicator_values = signal.generate(data)
            indicator_changes = indicator_values.pct_change()
            
            # Identify divergences
            bullish_div = (
                (price_changes < 0) &  # Price making lower lows
                (indicator_changes > 0) &  # Indicator making higher lows
                (indicator_changes.abs() > self._parameters['threshold'])
            )
            
            bearish_div = (
                (price_changes > 0) &  # Price making higher highs
                (indicator_changes < 0) &  # Indicator making lower highs
                (indicator_changes.abs() > self._parameters['threshold'])
            )
            
            # Confirm divergences over multiple periods
            confirmed_bull = bullish_div.rolling(
                window=self._parameters['confirmation']
            ).sum() >= self._parameters['confirmation']
            
            confirmed_bear = bearish_div.rolling(
                window=self._parameters['confirmation']
            ).sum() >= self._parameters['confirmation']
            
            # Generate signals
            decisions[confirmed_bull] = 1
            decisions[confirmed_bear] = -1
        
        return decisions

class AdaptiveStrategy(Strategy):
    """Strategy that adapts to market conditions."""
    
    def __init__(self,
                 name: str,
                 volatility_window: int = 20,
                 trend_threshold: float = 0.1):
        """
        Initialize adaptive strategy.
        
        Args:
            name: Strategy name
            volatility_window: Window for volatility calculation
            trend_threshold: Threshold for trend identification
        """
        super().__init__(name)
        self.set_parameters(
            vol_window=volatility_window,
            trend_threshold=trend_threshold
        )
    
    def evaluate(self, data: pd.DataFrame) -> pd.Series:
        """
        Evaluate strategy by adapting to market conditions.
        
        Returns:
            Series with strategy decisions (1 for buy, -1 for sell, 0 for hold)
        """
        if not self._signals:
            raise ValueError("No signals added to strategy")
        
        # Calculate market conditions
        returns = data['close'].pct_change()
        volatility = returns.rolling(
            window=self._parameters['vol_window']
        ).std()
        
        trend = returns.rolling(
            window=self._parameters['vol_window']
        ).mean()
        
        # Determine market regime
        is_trending = trend.abs() > self._parameters['trend_threshold']
        is_volatile = volatility > volatility.mean()
        
        # Generate base signals
        signals = pd.DataFrame()
        for signal in self._signals:
            signals[signal.name] = signal.generate(data)
        
        # Adapt strategy based on market conditions
        decisions = pd.Series(0, index=data.index)
        
        # In trending markets
        trending_signals = signals.mean(axis=1)
        decisions[is_trending] = np.sign(trending_signals[is_trending])
        
        # In volatile markets
        volatile_mask = is_volatile & ~is_trending
        decisions[volatile_mask] = signals.mean(axis=1)[volatile_mask]
        
        # Apply stronger threshold in volatile markets
        decisions[volatile_mask & (decisions.abs() < 0.5)] = 0
        
        return decisions.map(lambda x: int(np.sign(x)))

class FilteredStrategy(Strategy):
    """Strategy that filters signals based on multiple conditions."""
    
    def __init__(self,
                 name: str,
                 min_signals: int = 2,
                 volatility_filter: bool = True,
                 volume_filter: bool = True):
        """
        Initialize filtered strategy.
        
        Args:
            name: Strategy name
            min_signals: Minimum number of confirming signals
            volatility_filter: Whether to apply volatility filter
            volume_filter: Whether to apply volume filter
        """
        super().__init__(name)
        self.set_parameters(
            min_signals=min_signals,
            use_volatility=volatility_filter,
            use_volume=volume_filter,
            vol_window=20,
            vol_threshold=1.5,
            volume_threshold=1.5
        )
    
    def evaluate(self, data: pd.DataFrame) -> pd.Series:
        """
        Evaluate strategy with multiple filters.
        
        Returns:
            Series with strategy decisions (1 for buy, -1 for sell, 0 for hold)
        """
        if not self._signals:
            raise ValueError("No signals added to strategy")
        
        # Generate base signals
        signals = pd.DataFrame()
        for signal in self._signals:
            signals[signal.name] = signal.generate(data)
        
        # Count confirming signals
        bull_signals = (signals == 1).sum(axis=1)
        bear_signals = (signals == -1).sum(axis=1)
        
        # Initial decisions based on signal count
        decisions = pd.Series(0, index=data.index)
        min_sig = self._parameters['min_signals']
        
        decisions[bull_signals >= min_sig] = 1
        decisions[bear_signals >= min_sig] = -1
        
        # Apply volatility filter
        if self._parameters['use_volatility']:
            returns = data['close'].pct_change()
            volatility = returns.rolling(
                window=self._parameters['vol_window']
            ).std()
            vol_threshold = volatility.mean() * self._parameters['vol_threshold']
            decisions[volatility > vol_threshold] = 0
        
        # Apply volume filter
        if self._parameters['use_volume']:
            avg_volume = data['volume'].rolling(
                window=self._parameters['vol_window']
            ).mean()
            volume_threshold = avg_volume * self._parameters['volume_threshold']
            decisions[data['volume'] < volume_threshold] = 0
        
        return decisions

# Example usage of strategies:
"""
# Create indicators
sma_fast = MovingAverage(period=20)
sma_slow = MovingAverage(period=50)
rsi = RSI(period=14)
macd = MACD()
bb = BollingerBands()

# Create signals
crossover_signal = CrossoverSignal(sma_fast, sma_slow)
rsi_signal = RSISignal(rsi)
macd_signal = MACDSignal(macd)
bb_signal = BollingerBandsSignal(bb)

# Create and configure strategy
strategy = CompositeStrategy("MyStrategy")
strategy.add_signal(crossover_signal)
strategy.add_signal(rsi_signal)
strategy.add_signal(macd_signal)
strategy.add_signal(bb_signal)
strategy.set_parameters(weights={
    "Crossover": 0.3,
    "RSI": 0.2,
    "MACD": 0.3,
    "BB": 0.2
})

# Evaluate strategy
decisions = strategy.evaluate(data)
"""