from typing import Union, Optional
import numpy as np

class NumberFormatter:
    """Handles consistent number formatting across plots."""
    
    @staticmethod
    def format_currency(
        value: Union[float, int],
        currency: str = "$",
        precision: int = 2,
        abbreviate: bool = True
    ) -> str:
        """Format numbers as currency with optional abbreviation.
        
        Args:
            value: Number to format
            currency: Currency symbol
            precision: Decimal places to show
            abbreviate: Whether to abbreviate large numbers (K, M, B)
            
        Returns:
            Formatted currency string
        """
        pass
    
    @staticmethod
    def format_percentage(
        value: float,
        precision: int = 1,
        include_symbol: bool = True
    ) -> str:
        """Format numbers as percentages.
        
        Args:
            value: Number to format (0-100 or 0-1)
            precision: Decimal places to show
            include_symbol: Whether to include % symbol
            
        Returns:
            Formatted percentage string
        """
        pass
    
    @staticmethod
    def format_metric(
        value: float,
        precision: int = 2,
        prefix: str = "",
        suffix: str = "",
        abbreviate: bool = True
    ) -> str:
        """Format generic metric values.
        
        Args:
            value: Number to format
            precision: Decimal places to show
            prefix: Text to prepend
            suffix: Text to append
            abbreviate: Whether to abbreviate large numbers
            
        Returns:
            Formatted metric string
        """
        pass