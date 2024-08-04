#pyre-strict

from typing import Dict, Any, List
import pandas as pd
import matplotlib.pyplot as plt

class ModelRefreshPlots:
    """
    A class for generating various plots related to model refresh results.

    Methods:
    - __init__: Initialize the class with input and output data
    - actual_vs_fitted: Generate a plot comparing actual vs fitted values
    - decomposition_and_roi: Generate a stacked bar plot for decomposition and ROI
    - generate_all_plots: Generate all available plots
    """

    def __init__(self, input_collect_rf: Dict[str, Any], output_collect_rf: Dict[str, Any], 
                 report_collect: Dict[str, Any]) -> None:
        """
        Initialize the ModelRefreshPlots class with necessary data.

        :param input_collect_rf: Dictionary containing input data for refresh
        :param output_collect_rf: Dictionary containing output data for refresh
        :param report_collect: Dictionary containing report data
        """
        pass

    def actual_vs_fitted(self) -> Dict[str, Any]:
        """
        Generate a plot comparing actual vs fitted values for the model refresh.

        :return: Dictionary containing the plot and related data
        """
        pass

    def decomposition_and_roi(self) -> Dict[str, Any]:
        """
        Generate a stacked bar plot for decomposition and ROI across different refresh stages.

        :return: Dictionary containing the plot and related data
        """
        pass

    def generate_all_plots(self) -> Dict[str, Any]:
        """
        Generate all available plots for model refresh results.

        :return: Dictionary containing all generated plots and related data
        """
        pass
