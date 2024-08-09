from typing import Dict, Any, List

from robyn.data.entities.mmmdata_collection import MMMDataCollection
from robyn.modeling.entities.modeloutput_collection import ModelOutputCollection

class BudgetAllocationPlots:
    """
    A class for generating various plots related to budget allocation optimization.

    Methods:
    - __init__: Initialize the class with input and output data
    - response_spend_comparison: Generate a plot comparing total response and spend
    - channel_allocation_comparison: Generate a plot comparing allocation across channels
    - efficiency_metric_comparison: Generate a plot comparing efficiency metrics (ROAS or CPA)
    - marginal_efficiency_comparison: Generate a plot comparing marginal efficiency metrics
    - generate_all_plots: Generate all available plots
    """

    def __init__(self, mmmdata_collection: MMMDataCollection, modeloutput_collection: ModelOutputCollection, 
                 dt_optim_out: Dict[str, Any], select_model: str, scenario: str, 
                 eval_list: Dict[str, Any]) -> None:
        """
        Initialize the BudgetAllocationPlots class with necessary data.

        """
        pass

    def response_spend_comparison(self) -> Dict[str, Any]:
        """
        Generate a plot comparing total response and spend for initial, 
        bounded, and unbounded optimization results.

        :return: Dictionary containing the plot and related data
        """
        pass

    def channel_allocation_comparison(self) -> Dict[str, Any]:
        """
        Generate a plot comparing budget allocation across different channels 
        for initial, bounded, and unbounded optimization results.

        :return: Dictionary containing the plot and related data
        """
        pass

    def efficiency_metric_comparison(self) -> Dict[str, Any]:
        """
        Generate a plot comparing efficiency metrics (ROAS or CPA) 
        across channels for initial, bounded, and unbounded optimization results.

        :return: Dictionary containing the plot and related data
        """
        pass

    def marginal_efficiency_comparison(self) -> Dict[str, Any]:
        """
        Generate a plot comparing marginal efficiency metrics 
        across channels for initial, bounded, and unbounded optimization results.

        :return: Dictionary containing the plot and related data
        """
        pass

    def generate_all_plots(self) -> Dict[str, Any]:
        """
        Generate all available plots for budget allocation optimization.

        :return: Dictionary containing all generated plots and related data
        """
        pass
