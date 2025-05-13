# src/portfolio/optimization.py
"""
Portfolio optimization tools implementing various allocation strategies.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from scipy.optimize import minimize
import logging

logger = logging.getLogger(__name__)

class PortfolioOptimizer:
    """Portfolio optimization using various strategies."""
    
    def __init__(self, returns: pd.DataFrame):
        """
        Initialize portfolio optimizer.
        
        Args:
            returns: DataFrame of asset returns (each column is an asset)
        """
        self.returns = returns
        self.mean_returns = returns.mean()
        self.cov_matrix = returns.cov()
        self.assets = returns.columns.tolist()
        self.n_assets = len(self.assets)
        
    def minimum_variance(self,
                        constraints: Optional[Dict] = None) -> Dict[str, Union[float, np.ndarray]]:
        """
        Find the minimum variance portfolio.
        
        Args:
            constraints: Dictionary with optional constraints:
                - min_weight: Minimum weight per asset
                - max_weight: Maximum weight per asset
                - sum_weight: Target sum of weights (default 1.0)
                
        Returns:
            Dictionary containing:
                - weights: Optimal weights
                - volatility: Portfolio volatility
                - sharpe_ratio: Portfolio Sharpe ratio
        """
        constraints = constraints or {}
        min_weight = constraints.get('min_weight', 0.0)
        max_weight = constraints.get('max_weight', 1.0)
        sum_weight = constraints.get('sum_weight', 1.0)
        
        # Define constraints
        constraints_list = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - sum_weight}
        ]
        
        # Define bounds
        bounds = tuple((min_weight, max_weight) for _ in range(self.n_assets))
        
        # Initial guess
        x0 = np.array([1.0/self.n_assets] * self.n_assets)
        
        # Optimize
        result = minimize(
            lambda x: self._portfolio_volatility(x),
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list
        )
        
        if not result.success:
            logger.warning(f"Optimization failed: {result.message}")
        
        weights = result.x
        volatility = self._portfolio_volatility(weights)
        sharpe = self._portfolio_sharpe(weights)
        
        return {
            'weights': dict(zip(self.assets, weights)),
            'volatility': volatility,
            'sharpe_ratio': sharpe
        }
    
    def maximum_sharpe(self,
                      risk_free_rate: float = 0.0,
                      constraints: Optional[Dict] = None) -> Dict[str, Union[float, np.ndarray]]:
        """
        Find the maximum Sharpe ratio portfolio.
        
        Args:
            risk_free_rate: Risk-free rate
            constraints: Portfolio constraints
            
        Returns:
            Dictionary with optimal portfolio parameters
        """
        constraints = constraints or {}
        min_weight = constraints.get('min_weight', 0.0)
        max_weight = constraints.get('max_weight', 1.0)
        sum_weight = constraints.get('sum_weight', 1.0)
        
        constraints_list = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - sum_weight}
        ]
        
        bounds = tuple((min_weight, max_weight) for _ in range(self.n_assets))
        x0 = np.array([1.0/self.n_assets] * self.n_assets)
        
        result = minimize(
            lambda x: -self._portfolio_sharpe(x, risk_free_rate),
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list
        )
        
        if not result.success:
            logger.warning(f"Optimization failed: {result.message}")
        
        weights = result.x
        volatility = self._portfolio_volatility(weights)
        sharpe = self._portfolio_sharpe(weights, risk_free_rate)
        
        return {
            'weights': dict(zip(self.assets, weights)),
            'volatility': volatility,
            'sharpe_ratio': sharpe
        }
    
    def efficient_frontier(self,
                         n_points: int = 100,
                         constraints: Optional[Dict] = None) -> pd.DataFrame:
        """
        Generate efficient frontier points.
        
        Args:
            n_points: Number of points on the frontier
            constraints: Portfolio constraints
            
        Returns:
            DataFrame with returns and volatilities for frontier portfolios
        """
        min_ret = self.mean_returns.min() * 252
        max_ret = self.mean_returns.max() * 252
        target_returns = np.linspace(min_ret, max_ret, n_points)
        
        efficient_portfolios = []
        
        for target_return in target_returns:
            portfolio = self.efficient_return(
                target_return/252,  # Convert to daily return
                constraints
            )
            efficient_portfolios.append({
                'return': target_return,
                'volatility': portfolio['volatility'] * np.sqrt(252),
                'sharpe_ratio': portfolio['sharpe_ratio'],
                'weights': portfolio['weights']
            })
        
        return pd.DataFrame(efficient_portfolios)
    
    def efficient_return(self,
                        target_return: float,
                        constraints: Optional[Dict] = None) -> Dict[str, Union[float, np.ndarray]]:
        """
        Find the efficient portfolio for a target return.
        
        Args:
            target_return: Target portfolio return
            constraints: Portfolio constraints
            
        Returns:
            Dictionary with optimal portfolio parameters
        """
        constraints = constraints or {}
        min_weight = constraints.get('min_weight', 0.0)
        max_weight = constraints.get('max_weight', 1.0)
        sum_weight = constraints.get('sum_weight', 1.0)
        
        constraints_list = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - sum_weight},
            {'type': 'eq', 'fun': lambda x: self._portfolio_return(x) - target_return}
        ]
        
        bounds = tuple((min_weight, max_weight) for _ in range(self.n_assets))
        x0 = np.array([1.0/self.n_assets] * self.n_assets)
        
        result = minimize(
            lambda x: self._portfolio_volatility(x),
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list
        )
        
        if not result.success:
            logger.warning(f"Optimization failed: {result.message}")
        
        weights = result.x
        volatility = self._portfolio_volatility(weights)
        sharpe = self._portfolio_sharpe(weights)
        
        return {
            'weights': dict(zip(self.assets, weights)),
            'volatility': volatility,
            'sharpe_ratio': sharpe
        }
    
    def risk_parity(self,
                    risk_budget: Optional[List[float]] = None,
                    constraints: Optional[Dict] = None) -> Dict[str, Union[float, np.ndarray]]:
        """
        Implement risk parity portfolio optimization.
        
        Args:
            risk_budget: Target risk contribution for each asset
            constraints: Portfolio constraints
            
        Returns:
            Dictionary with optimal portfolio parameters
        """
        if risk_budget is None:
            risk_budget = [1.0/self.n_assets] * self.n_assets
            
        constraints = constraints or {}
        min_weight = constraints.get('min_weight', 0.0)
        max_weight = constraints.get('max_weight', 1.0)
        sum_weight = constraints.get('sum_weight', 1.0)
        
        def risk_budget_objective(x):
            portfolio_risk = self._portfolio_volatility(x)
            risk_contrib = self._risk_contribution(x)
            target_risk_contrib = portfolio_risk * np.array(risk_budget)
            return np.sum((risk_contrib - target_risk_contrib)**2)
        
        constraints_list = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - sum_weight}
        ]
        
        bounds = tuple((min_weight, max_weight) for _ in range(self.n_assets))
        x0 = np.array([1.0/self.n_assets] * self.n_assets)
        
        result = minimize(
            risk_budget_objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list
        )
        
        if not result.success:
            logger.warning(f"Optimization failed: {result.message}")
        
        weights = result.x
        volatility = self._portfolio_volatility(weights)
        sharpe = self._portfolio_sharpe(weights)
        
        return {
            'weights': dict(zip(self.assets, weights)),
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'risk_contribution': dict(zip(self.assets, self._risk_contribution(weights)))
        }
    
    def black_litterman(self,
                       market_caps: np.ndarray,
                       views: List[Dict],
                       tau: float = 0.025,
                       risk_free_rate: float = 0.0) -> Dict[str, Union[float, np.ndarray]]:
        """
        Implement Black-Litterman portfolio optimization.
        
        Args:
            market_caps: Market capitalizations of assets
            views: List of dictionaries containing views:
                - assets: List of asset names
                - weights: List of weights for the view
                - return: Expected return
                - confidence: Confidence level (0-1)
            tau: Weight on prior
            risk_free_rate: Risk-free rate
            
        Returns:
            Dictionary with optimal portfolio parameters
        """
        # Market weights
        mkt_weights = market_caps / np.sum(market_caps)
        
        # Prior expected returns (market equilibrium)
        pi = risk_free_rate + self._portfolio_volatility(mkt_weights) * mkt_weights
        
        # Create view matrix P and view vector q
        n_views = len(views)
        P = np.zeros((n_views, self.n_assets))
        q = np.zeros(n_views)
        omega = np.zeros((n_views, n_views))
        
        for i, view in enumerate(views):
            assets = view['assets']
            weights = view['weights']
            for asset, weight in zip(assets, weights):
                P[i, self.assets.index(asset)] = weight
            q[i] = view['return']
            omega[i, i] = (1 / view['confidence']) if view['confidence'] > 0 else 1e6
        
        # Compute posterior expected returns
        ts = tau * self.cov_matrix
        temp1 = np.linalg.inv(np.dot(np.dot(P, ts), P.T) + omega)
        temp2 = np.dot(np.dot(ts, P.T), temp1)
        posterior_returns = pi + np.dot(temp2, q - np.dot(P, pi))
        
        # Optimize with posterior returns
        posterior_optimizer = PortfolioOptimizer(
            pd.DataFrame(
                np.random.multivariate_normal(
                    posterior_returns,
                    self.cov_matrix,
                    size=252
                ),
                columns=self.assets
            )
        )
        
        return posterior_optimizer.maximum_sharpe(risk_free_rate)
    
    def _portfolio_return(self, weights: np.ndarray) -> float:
        """Calculate portfolio return."""
        return np.sum(self.mean_returns * weights)
    
    def _portfolio_volatility(self, weights: np.ndarray) -> float:
        """Calculate portfolio volatility."""
        return np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
    
    def _portfolio_sharpe(self,
                         weights: np.ndarray,
                         risk_free_rate: float = 0.0) -> float:
        """Calculate portfolio Sharpe ratio."""
        ret = self._portfolio_return(weights)
        vol = self._portfolio_volatility(weights)
        return (ret - risk_free_rate) / vol if vol > 0 else -np.inf
    
    def _risk_contribution(self, weights: np.ndarray) -> np.ndarray:
        """Calculate risk contribution of each asset."""
        port_vol = self._portfolio_volatility(weights)
        marginal_risk = np.dot(self.cov_matrix, weights)
        return weights * marginal_risk / port_vol if port_vol > 0 else np.zeros_like(weights)