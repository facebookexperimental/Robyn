# pyre-strict

from typing import Optional, Dict, Any
import pandas as pd
from robyn.data.entities.mmmdata import MMMData
from robyn.modeling.entities.modeloutputs import ModelOutputs


class ImmediateCarryoverCalculator:
    """
    A class to calculate immediate and carryover effects for media channels.

    This class handles the calculation of immediate and carryover effects for different
    media channels, including data preparation, decomposition, and result formatting.

    Attributes:
        input_collect (MMMData): Input data for the marketing mix model.
        output_collect (ModelOutputs): Output data from the model runs.
    """

    def __init__(self, input_collect: MMMData, output_collect: ModelOutputs):
        """
        Initialize the ImmediateCarryoverCalculator.

        Args:
            input_collect (MMMData): Input data for the marketing mix model.
            output_collect (ModelOutputs): Output data from the model runs.
        """
        self.input_collect = input_collect
        self.output_collect = output_collect

    def calculate(
        self, sol_id: Optional[str] = None, start_date: Optional[str] = None, end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Calculate immediate and carryover effects for media channels.

        Args:
            sol_id (Optional[str]): Solution ID to use. If None, uses the first solution.
            start_date (Optional[str]): Start date for the calculation. If None, uses the window start.
            end_date (Optional[str]): End date for the calculation. If None, uses the window end.

        Returns:
            pd.DataFrame: Dataframe with immediate and carryover effects.
        """
        # Implementation here
        pass

    def _get_date_range(self, start_date: Optional[str], end_date: Optional[str]) -> Dict[str, Any]:
        """
        Get the date range for the calculation.

        Args:
            start_date (Optional[str]): Start date for the calculation.
            end_date (Optional[str]): End date for the calculation.

        Returns:
            Dict[str, Any]: Dictionary containing start and end dates and indices.
        """
        # Implementation here
        pass

    def _calculate_saturated_dataframes(self, sol_id: str) -> Dict[str, pd.DataFrame]:
        """
        Calculate saturated dataframes with carryover and immediate parts.

        Args:
            sol_id (str): Solution ID to use.

        Returns:
            Dict[str, pd.DataFrame]: Dictionary of saturated dataframes.
        """
        # Implementation here
        pass

    def _calculate_decomposition(
        self, sol_id: str, saturated_dfs: Dict[str, pd.DataFrame], rolling_window: slice
    ) -> Dict[str, pd.DataFrame]:
        """
        Calculate decomposition of the effects.

        Args:
            sol_id (str): Solution ID to use.
            saturated_dfs (Dict[str, pd.DataFrame]): Saturated dataframes.
            rolling_window (slice): Slice object for the rolling window.

        Returns:
            Dict[str, pd.DataFrame]: Dictionary of decomposed effects.
        """
        # Implementation here
        pass

    def _prepare_result_dataframe(
        self, decomp_data: Dict[str, pd.DataFrame], start_date: str, end_date: str
    ) -> pd.DataFrame:
        """
        Prepare the final result dataframe.

        Args:
            decomp_data (Dict[str, pd.DataFrame]): Decomposed effects data.
            start_date (str): Start date of the calculation.
            end_date (str): End date of the calculation.

        Returns:
            pd.DataFrame: Final result dataframe with immediate and carryover effects.
        """
        # Implementation here
        pass
