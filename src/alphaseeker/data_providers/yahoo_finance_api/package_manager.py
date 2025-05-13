# src/yahoo_finance_api/__init__.py
"""
Package management utilities for the Yahoo Finance API.
"""

import logging
import subprocess
import sys
from typing import Dict

logger = logging.getLogger(__name__)

class PackageManager:
    """Handles package installation and checking."""
    
    @staticmethod
    def is_package_installed(package_name: str) -> bool:
        """
        Check if a package is installed.
        
        Args:
            package_name: Name of the package to check
            
        Returns:
            bool: True if package is installed, False otherwise
        """
        try:
            __import__(package_name)
            return True
        except ImportError:
            return False
    
    @staticmethod
    def install_package(package_name: str) -> bool:
        """
        Install a package using pip.
        
        Args:
            package_name: Name of the package to install
            
        Returns:
            bool: True if installation successful, False otherwise
        """
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install {package_name}: {str(e)}")
            return False

    @staticmethod
    def ensure_packages(required_packages: Dict[str, str]) -> None:
        """
        Ensure all required packages are installed.
        
        Args:
            required_packages: Dictionary mapping import names to pip package names
            
        Raises:
            ImportError: If package installation fails
        """
        for import_name, pip_name in required_packages.items():
            if not PackageManager.is_package_installed(import_name):
                logger.info(f"Installing required package: {pip_name}")
                if not PackageManager.install_package(pip_name):
                    raise ImportError(f"Failed to install required package: {pip_name}")