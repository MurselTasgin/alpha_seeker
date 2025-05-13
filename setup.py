# setup.py
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="alphaseeker",
    version="0.1.1",
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
        "PyPortfolioOpt>=1.5.6",
        
        # Date and time handling
        "python-dateutil>=2.9.0",
        "pytz>=2025.2",
        
        # Progress bars and logging
        "tqdm>=4.67.1",
        "colorlog>=6.9.0",
        
        # Financial calculations
        "empyrical>=0.5.5",  # Performance and risk metrics
    ],
    project_urls={
        "Bug Tracker": "https://github.com/MurselTasgin/alpha_seeker/issues",
        "Documentation": "https://alpha-seeker.readthedocs.io/",
        "Source Code": "https://github.com/MurselTasgin/alpha_seeker",
    },
)
