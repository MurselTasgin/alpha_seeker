# src/portfolio/risk.py
"""
Risk management tools for portfolio analysis and position sizing.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from scipy.stats import norm, t
import logging

logger = logging.getLogger(__name__)

class RiskManager:
    """Risk management and position sizing tools."""
    
    def __init__(self,
                 portfolio_value: float,
                 max_portfolio_risk: float = 0.02):
        """
        Initialize risk manager.
        
        Args:
            portfolio_value: Total portfolio value
            max_portfolio_risk: Maximum allowable portfolio risk (as decimal)
        """
        self.portfolio_value = portfolio_value
        self.max_portfolio_risk = max_portfolio_risk
        
    def calculate_position_size(self,
                              entry_price: float,
                              stop_loss: float,
                              risk_per_trade: float,
                              confidence: float = 0.95) -> Dict[str, float]:
        """
        Calculate position size based on fixed fractional position sizing.
        
        Args:
            entry_price: Entry price of the asset
            stop_loss: Stop loss price
            risk_per_trade: Risk amount per trade (as decimal)
            confidence: Confidence level for risk calculations
            
        Returns:
            Dictionary containing position details
        """
        # Calculate risk per share
        risk_per_share = abs(entry_price - stop_loss)
        if risk_per_share == 0:
            raise ValueError("Entry price cannot equal stop loss price")
        
        # Calculate maximum risk amount
        max_risk_amount = self.portfolio_value * risk_per_trade
        
        # Calculate position size
        position_size = max_risk_amount / risk_per_share
        position_value = position_size * entry_price
        
        return {
            'position_size': position_size,
            'position_value': position_value,
            'risk_amount': max_risk_amount,
            'risk_per_share': risk_per_share
        }
    
    def calculate_var(self,
                     returns: pd.Series,
                     confidence: float = 0.95,
                     method: str = 'historical') -> Dict[str, float]:
        """
        Calculate Value at Risk using various methods.
        
        Args:
            returns: Series of returns
            confidence: Confidence level
            method: VaR calculation method ('historical', 'parametric', 'monte_carlo')
            
        Returns:
            Dictionary with VaR calculations
        """
        if method == 'historical':
            var = -np.percentile(returns, (1 - confidence) * 100)
            cvar = -returns[returns <= -var].mean()
            
        elif method == 'parametric':
            z_score = norm.ppf(confidence)
            var = -(returns.mean() + z_score * returns.std())
            cvar = -(returns.mean() + 
                    returns.std() * norm.pdf(norm.ppf(1-confidence))/(1-confidence))
            
        elif method == 'monte_carlo':
            n_sims = 10000
            mu = returns.mean()
            sigma = returns.std()
            sim_returns = np.random.normal(mu, sigma, n_sims)
            var = -np.percentile(sim_returns, (1 - confidence) * 100)
            cvar = -sim_returns[sim_returns <= -var].mean()
            
        else:
            raise ValueError(f"Unknown VaR method: {method}")
        
        var_amount = self.portfolio_value * var
        cvar_amount = self.portfolio_value * cvar
        
        return {
            'var_pct': var,
            'cvar_pct': cvar,
            'var_amount': var_amount,
            'cvar_amount': cvar_amount
        }
    
    def kelly_criterion(self,
                       win_rate: float,
                       win_loss_ratio: float) -> Dict[str, float]:
        """
        Calculate optimal position size using Kelly Criterion.
        
        Args:
            win_rate: Probability of winning trade
            win_loss_ratio: Ratio of average win to average loss
            
        Returns:
            Dictionary with Kelly Criterion calculations
        """
        kelly_pct = win_rate - ((1 - win_rate) / win_loss_ratio)
        half_kelly = kelly_pct / 2  # More conservative estimate
        
        position_size_full = self.portfolio_value * max(0, kelly_pct)
        position_size_half = self.portfolio_value * max(0, half_kelly)
        
        return {
            'kelly_pct': kelly_pct,
            'half_kelly_pct': half_kelly,
            'position_size_full': position_size_full,
            'position_size_half': position_size_half
        }
    
    def risk_adjusted_position_size(self,
                                  returns: pd.Series,
                                  target_volatility: float) -> Dict[str, float]:
        """
        Calculate position size adjusted for volatility.
        
        Args:
            returns: Series of returns
            target_volatility: Target portfolio volatility
            
        Returns:
            Dictionary with position sizing details
        """
        current_vol = returns.std()
        vol_scalar = target_volatility / current_vol if current_vol > 0 else 0
        position_size = self.portfolio_value * vol_scalar
        
        return {
            'volatility_scalar': vol_scalar,
            'position_size': min(position_size, self.portfolio_value)
        }
    
    def drawdown_control(self,
                        returns: pd.Series,
                        max_drawdown: float = 0.1) -> Dict[str, Union[float, bool]]:
        """
        Implement drawdown-based position sizing control.
        
        Args:
            returns: Series of returns
            max_drawdown: Maximum allowable drawdown
            
        Returns:
            Dictionary with drawdown analysis and control signals
        """
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        current_drawdown = drawdown.iloc[-1]
        max_historical_drawdown = drawdown.min()
        
        # Calculate position size scalar based on drawdown
        drawdown_scalar = 1.0
        if current_drawdown < -max_drawdown:
            drawdown_scalar = 0.5  # Reduce position size by 50% when exceeding max drawdown
        
        return {
            'current_drawdown': current_drawdown,
            'max_historical_drawdown': max_historical_drawdown,
            'position_scalar': drawdown_scalar,
            'reduce_exposure': current_drawdown < -max_drawdown
        }
    
    def correlation_based_limits(self,
                               returns: pd.DataFrame,
                               max_correlation: float = 0.7) -> Dict[str, Union[pd.DataFrame, List[str]]]:
        """
        Implement correlation-based position limits.
        
        Args:
            returns: DataFrame of asset returns
            max_correlation: Maximum allowable correlation between assets
            
        Returns:
            Dictionary with correlation analysis and recommendations
        """
        corr_matrix = returns.corr()
        
        # Identify highly correlated pairs
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > max_correlation:
                    high_corr_pairs.append(
                        (corr_matrix.columns[i],
                         corr_matrix.columns[j],
                         corr_matrix.iloc[i, j])
                    )
        
        # Generate recommendations
        reduce_exposure = []
        for asset1, asset2, corr in high_corr_pairs:
            # Recommend reducing exposure in the asset with lower Sharpe ratio
            sharpe1 = returns[asset1].mean() / returns[asset1].std()
            sharpe2 = returns[asset2].mean() / returns[asset2].std()
            reduce_exposure.append(asset1 if sharpe1 < sharpe2 else asset2)
        
        return {
            'correlation_matrix': corr_matrix,
            'high_correlation_pairs': high_corr_pairs,
            'reduce_exposure': list(set(reduce_exposure))
        }
    
    def stress_test(self,
                   returns: pd.Series,
                   scenarios: Dict[str, float]) -> Dict[str, float]:
        """
        Perform stress testing on the portfolio.
        
        Args:
            returns: Series of returns
            scenarios: Dictionary of stress scenarios and their shock sizes
            
        Returns:
            Dictionary with stress test results
        """
        results = {}
        
        for scenario, shock in scenarios.items():
            # Calculate impact on portfolio value
            impact = self.portfolio_value * (1 + shock) - self.portfolio_value
            
            # Calculate maximum drawdown under stress
            stress_returns = returns + shock
            cum_returns = (1 + stress_returns).cumprod()
            running_max = cum_returns.expanding().max()
            drawdown = (cum_returns - running_max) / running_max
            max_drawdown = drawdown.min()
            
            results[scenario] = {
                'impact': impact,
                'final_value': self.portfolio_value * (1 + shock),
                'max_drawdown': max_drawdown
            }
        
        return results
    
    def risk_report(self,
                   returns: pd.Series,
                   positions: Dict[str, float]) -> Dict[str, Any]:
        """
        Generate comprehensive risk report.
        
        Args:
            returns: Series of returns
            positions: Dictionary of current positions and their values
            
        Returns:
            Dictionary with risk analysis
        """
        var_calc = self.calculate_var(returns)
        
        # Calculate various risk metrics
        volatility = returns.std() * np.sqrt(252)  # Annualized volatility
        sharpe = (returns.mean() * 252) / volatility if volatility > 0 else 0
        sortino = (returns.mean() * 252) / (returns[returns < 0].std() * np.sqrt(252))
        
        # Calculate drawdown metrics
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max
        current_drawdown = drawdown.iloc[-1]
        max_drawdown = drawdown.min()
        
        # Position concentration
        total_exposure = sum(abs(v) for v in positions.values())
        concentration = {k: abs(v)/total_exposure for k, v in positions.items()}
        
        return {
            'var_metrics': var_calc,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'drawdown_metrics': {
                'current': current_drawdown,
                'maximum': max_drawdown
            },
            'position_concentration': concentration,
            'total_exposure': total_exposure,
            'exposure_to_equity_ratio': total_exposure / self.portfolio_value,
            'risk_capacity_used': total_exposure * volatility / self.max_portfolio_risk
        }