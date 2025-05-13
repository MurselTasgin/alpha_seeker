# src/technical_indicators/performance.py
"""
Performance analysis tools for strategy evaluation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

class PerformanceAnalyzer:
    """Analyzes trading strategy performance."""
    
    def __init__(self, data: pd.DataFrame, signals: pd.Series):
        """
        Initialize performance analyzer.
        
        Args:
            data: Price/volume data
            signals: Strategy signals (1 for buy, -1 for sell, 0 for hold)
        """
        self.data = data
        self.signals = signals
        self._results: Dict[str, Any] = {}
    
    def calculate_returns(self,
                        commission: float = 0.001,
                        slippage: float = 0.001) -> pd.Series:
        """
        Calculate strategy returns including transaction costs.
        
        Args:
            commission: Commission rate per trade
            slippage: Slippage rate per trade
            
        Returns:
            Series of strategy returns
        """
        # Calculate position changes
        position_changes = self.signals.diff().fillna(0)
        
        # Calculate transaction costs
        transaction_costs = (
            (abs(position_changes) * commission) +
            (abs(position_changes) * slippage)
        )
        
        # Calculate price returns
        price_returns = self.data['close'].pct_change()
        
        # Calculate strategy returns
        strategy_returns = (
            self.signals.shift(1) * price_returns -
            transaction_costs
        )
        
        self._results['returns'] = strategy_returns
        return strategy_returns
    
    def calculate_metrics(self) -> Dict[str, float]:
        """
        Calculate performance metrics.
        
        Returns:
            Dictionary of performance metrics
        """
        if 'returns' not in self._results:
            self.calculate_returns()
            
        returns = self._results['returns']
        
        # Basic metrics
        total_return = (1 + returns).prod() - 1
        annual_return = (1 + returns).prod() ** (252/len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility != 0 else 0
        
        # Drawdown analysis
        cum_returns = (1 + returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdowns = cum_returns / rolling_max - 1
        max_drawdown = drawdowns.min()
        
        # Trading metrics
        trades = self.signals.diff().fillna(0)
        n_trades = (trades != 0).sum()
        win_rate = (returns[trades != 0] > 0).mean()
        
        # Profit metrics
        profit_factor = abs(returns[returns > 0].sum() / returns[returns < 0].sum()) \
            if returns[returns < 0].sum() != 0 else np.inf
        
        self._results['metrics'] = {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'number_of_trades': n_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor
        }
        
        return self._results['metrics']
    
    def get_trade_details(self) -> pd.DataFrame:
        """
        Get detailed trade statistics.
        
        Returns:
            DataFrame with trade details
        """
        position_changes = self.signals.diff().fillna(0)
        trade_points = position_changes[position_changes != 0]
        
        trades = []
        current_position = 0
        entry_price = 0
        entry_time = None
        
        for time, change in trade_points.items():
            price = self.data.loc[time, 'close']
            
            if current_position == 0:  # Opening trade
                current_position = change
                entry_price = price
                entry_time = time
            else:  # Closing trade
                pnl = (price - entry_price) * current_position
                pnl_pct = (price / entry_price - 1) * current_position
                duration = (time - entry_time).days
                
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': time,
                    'duration': duration,
                    'direction': 'long' if current_position > 0 else 'short',
                    'entry_price': entry_price,
                    'exit_price': price,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct
                })
                
                current_position = 0
        
        if trades:
            trades_df = pd.DataFrame(trades)
            self._results['trades'] = trades_df
            return trades_df
        return pd.DataFrame()
    
    def plot_equity_curve(self) -> 'go.Figure':
        """
        Plot equity curve with drawdowns.
        
        Returns:
            Plotly figure object
        """
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        if 'returns' not in self._results:
            self.calculate_returns()
            
        returns = self._results['returns']
        cum_returns = (1 + returns).cumprod()
        
        # Calculate drawdowns
        rolling_max = cum_returns.expanding().max()
        drawdowns = (cum_returns / rolling_max - 1) * 100
        
        # Create figure
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.7, 0.3]
        )
        
        # Add equity curve
        fig.add_trace(
            go.Scatter(
                x=cum_returns.index,
                y=cum_returns,
                name='Equity Curve',
                line=dict(color='blue')
            ),
            row=1, col=1
        )
        
        # Add drawdown
        fig.add_trace(
            go.Scatter(
                x=drawdowns.index,
                y=drawdowns,
                name='Drawdown',
                fill='tozeroy',
                line=dict(color='red')
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            title='Strategy Equity Curve and Drawdown',
            xaxis2_title='Date',
            yaxis_title='Equity',
            yaxis2_title='Drawdown %',
            showlegend=True,
            height=800
        )
        
        return fig
    
    def plot_monthly_returns(self) -> 'go.Figure':
        """
        Plot monthly returns heatmap.
        
        Returns:
            Plotly figure object
        """
        import plotly.graph_objects as go
        
        if 'returns' not in self._results:
            self.calculate_returns()
            
        # Calculate monthly returns
        returns = self._results['returns']
        monthly_returns = returns.resample('M').agg(lambda x: (1 + x).prod() - 1)
        
        # Create monthly returns matrix
        returns_matrix = pd.DataFrame()
        for year in monthly_returns.index.year.unique():
            year_returns = monthly_returns[monthly_returns.index.year == year]
            returns_matrix[year] = year_returns.values
            
        returns_matrix.index = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=returns_matrix.values * 100,
            x=returns_matrix.columns,
            y=returns_matrix.index,
            colorscale='RdYlGn',
            text=np.round(returns_matrix.values * 100, 1),
            texttemplate='%{text:.1f}%',
            textfont={"size": 10},
            colorbar=dict(title='Returns %')
        ))
        
        fig.update_layout(
            title='Monthly Returns Heatmap',
            xaxis_title='Year',
            yaxis_title='Month',
            height=500
        )
        
        return fig
    
    def generate_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive performance report.
        
        Returns:
            Dictionary containing all performance metrics and analyses
        """
        # Calculate all metrics if not already calculated
        if 'metrics' not in self._results:
            self.calculate_metrics()
        
        if 'trades' not in self._results:
            self.get_trade_details()
        
        # Additional analyses
        trades_df = self._results['trades']
        returns = self._results['returns']
        
        # Monthly analysis
        monthly_returns = returns.resample('M').agg(lambda x: (1 + x).prod() - 1)
        best_month = monthly_returns.max()
        worst_month = monthly_returns.min()
        
        # Streak analysis
        winning_streak = self._calculate_streak(returns > 0)
        losing_streak = self._calculate_streak(returns < 0)
        
        # Risk metrics
        var_95 = returns.quantile(0.05)
        var_99 = returns.quantile(0.01)
        
        # Combine all results
        report = {
            'summary_metrics': self._results['metrics'],
            'monthly_stats': {
                'best_month': best_month,
                'worst_month': worst_month,
                'average_month': monthly_returns.mean(),
                'monthly_std': monthly_returns.std()
            },
            'streak_analysis': {
                'longest_winning_streak': winning_streak,
                'longest_losing_streak': losing_streak
            },
            'risk_metrics': {
                'var_95': var_95,
                'var_99': var_99,
                'skewness': returns.skew(),
                'kurtosis': returns.kurtosis()
            },
            'trade_analysis': {
                'avg_trade_duration': trades_df['duration'].mean(),
                'avg_profit_per_trade': trades_df['pnl'].mean(),
                'largest_winner': trades_df['pnl'].max(),
                'largest_loser': trades_df['pnl'].min(),
                'avg_win': trades_df[trades_df['pnl'] > 0]['pnl'].mean(),
                'avg_loss': trades_df[trades_df['pnl'] < 0]['pnl'].mean()
            }
        }
        
        return report
    
    @staticmethod
    def _calculate_streak(condition: pd.Series) -> int:
        """Calculate longest streak of True values."""
        streak = 0
        max_streak = 0
        
        for value in condition:
            if value:
                streak += 1
                max_streak = max(max_streak, streak)
            else:
                streak = 0
                
        return max_streak