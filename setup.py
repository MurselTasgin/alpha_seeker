# setup.py
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="alpha_seeker",
    version="0.1.0",
    author="Mursel Tasgin",
    author_email="mursel.tasgin@gmail.com",
    description="A modern Stock and ETF analysis package with Yahoo Finance API wrapper with automatic dependency management",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MurselTasgin/alpha_seeker",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.10",
    install_requires=[
        # Core dependencies
        "yfinance==0.2.60",
        "pandas>=2.2.2",
        "curl-cffi>=0.10.0",
        
        # Data manipulation and analysis
        "numpy>=2.2.5",
        "scipy>=1.15.3",
        
        # Technical Analysis
        "ta>=0.11.0",  # Technical analysis indicators
        
        # Visualization
        "plotly>=5.0.0",
        "matplotlib>=3.10.3",
        "seaborn>=0.13.0",
        
        # Statistics and Machine Learning
        "statsmodels>=0.14.4",
        "scikit-learn>=1.6.0",
        
        # Optimization
        "cvxopt>=1.3.0",  # For portfolio optimization
        ###"pypfopt>=1.4.1,<1.5.0",  # Portfolio optimization utilities
        "PyPortfolioOpt>=1.5.6",
        
        # Date and time handling
        "python-dateutil>=2.9.0",
        "pytz>=2025.2",
        
        # Progress bars and logging
        "tqdm>=4.67.1",
        "colorlog>=6.9.0",
        
        # Financial calculations
        "empyrical>=0.5.5",  # Performance and risk metrics
        ###"risk-kit>=1.0.0",   # Risk analysis tools
    ],
    extras_require={
        # Development dependencies
        "dev": [
            "pytest>=8.3",
            "pytest-cov>=4.0",
            "black>=22.0",
            "isort>=5.0",
            "flake8>=4.0",
            "mypy>=0.910",
            "pylint>=2.10.0",
            "jupyter>=1.0.0",
        ],
        
        # Documentation dependencies
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "sphinx-autodoc-typehints>=1.12.0",
            "nbsphinx>=0.8.0",
        ],
        
        # Extra analysis tools
        "analysis": [
            "arch>=5.0.0",        # ARCH models for volatility
            "pmdarima>=2.0.0",    # Auto ARIMA modeling
            "prophet>=1.1.0",     # Facebook Prophet forecasting
            "keras>=2.6.0",       # Deep learning support
            "tensorflow>=2.6.0",   # Deep learning support
        ],
        
        # Extra optimization tools
        "optimize": [
            "pulp>=2.5.0",        # Linear programming
            "pyomo>=6.1.0",       # Optimization modeling
            "gekko>=1.0.0",       # Advanced optimization
        ],
        
        # Data acquisition extras
        "data": [
            "pandas-datareader>=0.10.0",  # Additional data sources
            "alpha_vantage>=2.3.1",       # Alpha Vantage API
            "finnhub-python>=2.4.0",      # Finnhub API
            "polygon-api-client>=1.0.0",   # Polygon.io API
        ],
    },
    project_urls={
        "Bug Tracker": "https://github.com/MurselTasgin/alpha_seeker/issues",
        "Documentation": "https://alpha-seeker.readthedocs.io/",
        "Source Code": "https://github.com/MurselTasgin/alpha_seeker",
    },
)
