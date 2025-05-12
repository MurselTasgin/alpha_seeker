"""
Plotting utilities for Yahoo Finance API data visualization.
"""

import logging
from typing import Optional, List, Dict, Any, Union
from datetime import datetime, timedelta

from yahoo_finance_api.package_manager import PackageManager

logger = logging.getLogger(__name__)

class StockPlotter:
    """A class for creating interactive financial charts using Plotly."""
    
    # Required packages for plotting
    REQUIRED_PACKAGES = {
        'plotly': 'plotly',
        'pandas': 'pandas',
        'numpy': 'numpy'
    }
    
    def __init__(self):
        """Initialize the StockPlotter with required packages."""
        # Ensure required packages are installed
        PackageManager.ensure_packages(self.REQUIRED_PACKAGES)
        
        # Import required packages
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            import pandas as pd
            import numpy as np
            
            self.go = go
            self.make_subplots = make_subplots
            self.pd = pd
            self.np = np
            
        except ImportError as e:
            raise ImportError(f"Failed to import required packages: {str(e)}")

    def _calculate_technical_indicators(self, df: 'pd.DataFrame') -> 'pd.DataFrame':
        """
        Calculate technical indicators for the dataset.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with additional technical indicators
        """
        # Moving averages
        df['MA20'] = df['close'].rolling(window=20).mean()
        df['MA50'] = df['close'].rolling(window=50).mean()
        df['MA200'] = df['close'].rolling(window=200).mean()
        
        # Bollinger Bands
        df['BB_middle'] = df['close'].rolling(window=20).mean()
        df['BB_upper'] = df['BB_middle'] + 2*df['close'].rolling(window=20).std()
        df['BB_lower'] = df['BB_middle'] - 2*df['close'].rolling(window=20).std()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Histogram'] = df['MACD'] - df['Signal_Line']
        
        return df

    def plot_candlestick(self, 
                        df: 'pd.DataFrame',
                        title: str = "",
                        indicators: List[str] = None,
                        volume: bool = True,
                        show_grid: bool = True,
                        theme: str = "light",
                        width: int = 1200,
                        height: int = 800) -> 'go.Figure':
        """
        Create an interactive candlestick chart with optional indicators.
        
        Args:
            df: DataFrame with OHLCV data
            title: Chart title
            indicators: List of indicators to show ('MA', 'BB', 'RSI', 'MACD')
            volume: Whether to show volume
            show_grid: Whether to show grid
            theme: Chart theme ('light' or 'dark')
            width: Chart width in pixels
            height: Chart height in pixels
            
        Returns:
            Plotly figure object
        """
        indicators = indicators or []
        rows = 1 + ('RSI' in indicators) + ('MACD' in indicators)
        
        # Create figure with secondary y-axis for volume
        fig = self.make_subplots(
            rows=rows,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.6] + [0.2] * (rows-1),
            specs=[[{"secondary_y": True}]] + [[{"secondary_y": False}]] * (rows-1)
        )

        # Add candlestick trace
        fig.add_trace(
            self.go.Candlestick(
                x=df['date'],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='OHLC',
                increasing_line_color='green',
                decreasing_line_color='red'
            ),
            row=1, col=1, secondary_y=False
        )

        # Add volume bars
        if volume:
            colors = ['green' if row['close'] >= row['open'] else 'red' 
                     for _, row in df.iterrows()]
            
            fig.add_trace(
                self.go.Bar(
                    x=df['date'],
                    y=df['volume'],
                    marker_color=colors,
                    name='Volume',
                    opacity=0.5
                ),
                row=1, col=1, secondary_y=True
            )

        # Calculate and add technical indicators
        df = self._calculate_technical_indicators(df)
        
        # Add moving averages
        if 'MA' in indicators:
            fig.add_trace(
                self.go.Scatter(
                    x=df['date'], y=df['MA20'],
                    name='MA20', line=dict(color='blue', width=1)
                ),
                row=1, col=1, secondary_y=False
            )
            fig.add_trace(
                self.go.Scatter(
                    x=df['date'], y=df['MA50'],
                    name='MA50', line=dict(color='orange', width=1)
                ),
                row=1, col=1, secondary_y=False
            )
            fig.add_trace(
                self.go.Scatter(
                    x=df['date'], y=df['MA200'],
                    name='MA200', line=dict(color='purple', width=1)
                ),
                row=1, col=1, secondary_y=False
            )

        # Add Bollinger Bands
        if 'BB' in indicators:
            fig.add_trace(
                self.go.Scatter(
                    x=df['date'], y=df['BB_upper'],
                    name='BB Upper', line=dict(color='gray', width=1)
                ),
                row=1, col=1, secondary_y=False
            )
            fig.add_trace(
                self.go.Scatter(
                    x=df['date'], y=df['BB_middle'],
                    name='BB Middle', line=dict(color='gray', width=1)
                ),
                row=1, col=1, secondary_y=False
            )
            fig.add_trace(
                self.go.Scatter(
                    x=df['date'], y=df['BB_lower'],
                    name='BB Lower', line=dict(color='gray', width=1)
                ),
                row=1, col=1, secondary_y=False
            )

        # Add RSI
        if 'RSI' in indicators:
            current_row = 2
            fig.add_trace(
                self.go.Scatter(
                    x=df['date'], y=df['RSI'],
                    name='RSI', line=dict(color='purple', width=1)
                ),
                row=current_row, col=1
            )
            # Add RSI levels
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=current_row)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=current_row)

        # Add MACD
        if 'MACD' in indicators:
            current_row = 2 if 'RSI' not in indicators else 3
            fig.add_trace(
                self.go.Scatter(
                    x=df['date'], y=df['MACD'],
                    name='MACD', line=dict(color='blue', width=1)
                ),
                row=current_row, col=1
            )
            fig.add_trace(
                self.go.Scatter(
                    x=df['date'], y=df['Signal_Line'],
                    name='Signal Line', line=dict(color='orange', width=1)
                ),
                row=current_row, col=1
            )
            fig.add_trace(
                self.go.Bar(
                    x=df['date'], y=df['MACD_Histogram'],
                    name='MACD Histogram',
                    marker_color=['green' if val >= 0 else 'red' for val in df['MACD_Histogram']]
                ),
                row=current_row, col=1
            )

        # Update layout
        fig.update_layout(
            title=title,
            xaxis_rangeslider_visible=False,
            template='plotly_dark' if theme == 'dark' else 'plotly_white',
            showlegend=True,
            hovermode='x unified',
            width=width,
            height=height
        )

        # Update y-axes
        fig.update_yaxes(title_text="Price", row=1, col=1, secondary_y=False)
        if volume:
            fig.update_yaxes(
                title_text="Volume",
                row=1, col=1,
                secondary_y=True,
                showgrid=False
            )
        if 'RSI' in indicators:
            fig.update_yaxes(title_text="RSI", row=2, col=1)
        if 'MACD' in indicators:
            fig.update_yaxes(
                title_text="MACD",
                row=2 if 'RSI' not in indicators else 3,
                col=1
            )

        # Update grid
        fig.update_xaxes(showgrid=show_grid)
        fig.update_yaxes(showgrid=show_grid)

        return fig

    def plot_comparison(self,
                       dfs: List['pd.DataFrame'],
                       symbols: List[str],
                       title: str = "Price Comparison",
                       normalize: bool = True,
                       theme: str = "light",
                       width: int = 1200,
                       height: int = 600) -> 'go.Figure':
        """
        Create a comparison chart for multiple symbols.
        
        Args:
            dfs: List of DataFrames with price data
            symbols: List of symbol names
            title: Chart title
            normalize: Whether to normalize prices to percentage change
            theme: Chart theme ('light' or 'dark')
            width: Chart width in pixels
            height: Chart height in pixels
            
        Returns:
            Plotly figure object
        """
        fig = self.go.Figure()

        for df, symbol in zip(dfs, symbols):
            y = df['close']
            if normalize:
                # Convert to percentage change from first day
                y = (y / y.iloc[0] - 1) * 100

            fig.add_trace(
                self.go.Scatter(
                    x=df['date'],
                    y=y,
                    name=symbol,
                    mode='lines'
                )
            )

        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="% Change" if normalize else "Price",
            template='plotly_dark' if theme == 'dark' else 'plotly_white',
            showlegend=True,
            hovermode='x unified',
            width=width,
            height=height
        )

        return fig

    def plot_correlation_matrix(self,
                              dfs: List['pd.DataFrame'],
                              symbols: List[str],
                              title: str = "Correlation Matrix",
                              theme: str = "light",
                              width: int = 800,
                              height: int = 800) -> 'go.Figure':
        """
        Create a correlation matrix heatmap for multiple symbols.
        
        Args:
            dfs: List of DataFrames with price data
            symbols: List of symbol names
            title: Chart title
            theme: Chart theme ('light' or 'dark')
            width: Chart width in pixels
            height: Chart height in pixels
            
        Returns:
            Plotly figure object
        """
        # Calculate returns
        returns = self.pd.DataFrame()
        for df, symbol in zip(dfs, symbols):
            returns[symbol] = df['close'].pct_change()

        # Calculate correlation matrix
        corr_matrix = returns.corr()

        # Create heatmap
        fig = self.go.Figure(data=self.go.Heatmap(
            z=corr_matrix,
            x=symbols,
            y=symbols,
            zmin=-1,
            zmax=1,
            text=corr_matrix.round(2),
            texttemplate='%{text}',
            textfont={"size": 10},
            colorscale='RdBu'
        ))

        fig.update_layout(
            title=title,
            template='plotly_dark' if theme == 'dark' else 'plotly_white',
            width=width,
            height=height
        )

        return fig