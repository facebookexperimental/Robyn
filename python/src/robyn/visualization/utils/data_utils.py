import pandas as pd
from typing import List, Optional

class DataPreprocessor:
    """Handles common data preprocessing for plots."""
    
    @staticmethod
    def calculate_moving_average(
        data: pd.Series,
        window: int = 7,
        min_periods: Optional[int] = None
    ) -> pd.Series:
        """Calculate moving average of time series data.
        
        Args:
            data: Time series data
            window: Rolling window size
            min_periods: Minimum periods required for calculation
            
        Returns:
            Series with moving averages
        """
        pass
    
    @staticmethod
    def normalize_series(
        data: pd.Series,
        method: str = 'minmax'
    ) -> pd.Series:
        """Normalize data series to 0-1 range.
        
        Args:
            data: Data to normalize
            method: Normalization method ('minmax' or 'standard')
            
        Returns:
            Normalized data series
        """
        pass
    
    @staticmethod
    def detect_outliers(
        data: pd.Series,
        method: str = 'iqr',
        threshold: float = 1.5
    ) -> pd.Series:
        """Identify outliers in data series.
        
        Args:
            data: Data to check for outliers
            method: Detection method ('iqr' or 'zscore')
            threshold: Outlier threshold value
            
        Returns:
            Boolean series indicating outliers
        """
        pass