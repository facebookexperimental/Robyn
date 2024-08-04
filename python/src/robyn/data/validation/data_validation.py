import pandas as pd
import numpy as np
from typing import List, Dict, Any

class DataValidation:
    def __init__(self):
        pass

    def check_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Check for missing values in the data.

        Args:
            data (pd.DataFrame): The data to check.

        Returns:
            pd.DataFrame: A DataFrame indicating the count of missing values per column.
        """
        missing_values = data.isnull().sum()
        return missing_values[missing_values > 0].to_frame(name='missing_count')

    def check_duplicates(self, data: pd.DataFrame) -> int:
        """
        Check for duplicate rows in the data.

        Args:
            data (pd.DataFrame): The data to check.

        Returns:
            int: The number of duplicate rows.
        """
        return data.duplicated().sum()

    def check_outliers(self, data: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
        """
        Detect outliers in the data using Z-score.

        Args:
            data (pd.DataFrame): The data to check.
            threshold (float): The Z-score threshold to identify outliers.

        Returns:
            pd.DataFrame: A DataFrame indicating the count of outliers per column.
        """
        numeric_data = data.select_dtypes(include=[np.number])
        z_scores = np.abs((numeric_data - numeric_data.mean()) / numeric_data.std())
        outliers = (z_scores > threshold).sum()
        return outliers[outliers > 0].to_frame(name='outlier_count')

    def check_consistency(self, data: pd.DataFrame, columns: List[str]) -> Dict[str, bool]:
        """
        Check consistency of specified columns in the data.

        Args:
            data (pd.DataFrame): The data to check.
            columns (List[str]): The columns to check for consistency.

        Returns:
            Dict[str, bool]: A dictionary indicating whether each column is consistent.
        """
        consistency_results = {}
        for column in columns:
            unique_values = data[column].nunique()
            consistency_results[column] = unique_values == 1
        return consistency_results

    def check_completeness(self, data: pd.DataFrame, required_columns: List[str]) -> Dict[str, bool]:
        """
        Check if all required columns are present in the data.

        Args:
            data (pd.DataFrame): The data to check.
            required_columns (List[str]): The list of required columns.

        Returns:
            Dict[str, bool]: A dictionary indicating whether each required column is present.
        """
        completeness_results = {}
        for column in required_columns:
            completeness_results[column] = column in data.columns
        return completeness_results

    def validate_data(self, data: pd.DataFrame, required_columns: List[str], consistency_columns: List[str]) -> Dict[str, Any]:
        """
        Perform all validation checks on the data.

        Args:
            data (pd.DataFrame): The data to validate.
            required_columns (List[str]): The list of required columns.
            consistency_columns (List[str]): The columns to check for consistency.

        Returns:
            Dict[str, Any]: A dictionary containing validation results.
        """
        validation_results = {
            'missing_values': self.check_missing_values(data),
            'duplicates': self.check_duplicates(data),
            'outliers': self.check_outliers(data),
            'consistency': self.check_consistency(data, consistency_columns),
            'completeness': self.check_completeness(data, required_columns),
        }
        return validation_results

# Example usage
if __name__ == "__main__":
    # Sample data for demonstration
    sample_data = pd.DataFrame({
        'date': pd.date_range(start='2023-01-01', periods=10, freq='D'),
        'spend': [100, 200, np.nan, 400, 500, 600, 700, 800, 900, 1000],
        'impressions': [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000],
        'channel': ['facebook', 'facebook', 'facebook', 'facebook', 'facebook', 'facebook', 'facebook', 'facebook', 'facebook', 'facebook']
    })

    # Instantiate the DataValidation class
    validator = DataValidation()

    # Perform all validation checks
    results = validator.validate_data(
        sample_data,
        required_columns=['date', 'spend', 'impressions', 'channel'],
        consistency_columns=['channel']
    )

    # Display validation results
    for key, value in results.items():
        print(f"{key}:\n{value}\n")
