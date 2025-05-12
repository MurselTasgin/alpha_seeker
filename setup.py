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
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.7",
    install_requires=[
        "yfinance==0.2.60",
        "pandas>=1.0.0",
        "curl-cffi>=0.5.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=22.0",
            "isort>=5.0",
            "flake8>=4.0",
        ],
    },
)