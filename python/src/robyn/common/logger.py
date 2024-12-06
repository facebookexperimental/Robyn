# pyre-strict

import logging

import pandas as pd


class RobynLogger:

    @staticmethod
    def log_df(
        logger,
        df: pd.DataFrame,
        logLevel: int = logging.DEBUG,
        print_head: bool = False,
    ):
        """
        Log the shape and first few rows of a DataFrame.

        Args:
            df (pd.DataFrame): DataFrame to log.
            name (str): Name of the DataFrame.
        """
        if df is None:
            logger.log(logLevel, "DataFrame is None")
            return

        logger.log(logLevel, f"DataFrame columns: {df.columns}")
        logger.log(logLevel, f"DataFrame Shape: {df.shape}")
        if print_head:
            logger.log(logLevel, f"DataFrame Head: {df.head()}")
