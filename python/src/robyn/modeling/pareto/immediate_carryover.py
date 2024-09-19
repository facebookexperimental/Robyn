# pyre-strict

from typing import Optional, Dict, List
from dataclasses import dataclass
import pandas as pd
import numpy as np
from datetime import datetime
from robyn.data.entities.mmmdata import MMMData
from robyn.modeling.entities.modeloutputs import ModelOutputs


@dataclass
class DateRange:
    """
    Represents a date range for calculations.

    Attributes:
        start_date (datetime): Start date of the range.
        end_date (datetime): End date of the range.
        start_index (int): Index corresponding to the start date in the data.
        end_index (int): Index corresponding to the end date in the data.
    """

    start_date: datetime
    end_date: datetime
    start_index: int
    end_index: int


@dataclass
class EffectDecomposition:
    """
    Represents the decomposition of effects for a channel.

    Attributes:
        channel (str): Name of the channel.
        immediate_effect (float): Immediate effect value.
        carryover_effect (float): Carryover effect value.
        total_effect (float): Total effect (immediate + carryover).
    """

    channel: str
    immediate_effect: float
    carryover_effect: float
    total_effect: float


class ImmediateCarryoverCalculator:
    """
    Calculates immediate and carryover effects for media channels in marketing mix models.

    This class handles the calculation of immediate and carryover effects for different
    media channels, including data preparation, decomposition, and result aggregation.

    Attributes:
        mmm_data (MMMData): Input data for the marketing mix model.
        model_outputs (ModelOutputs): Output data from the model runs.
    """

    def __init__(self, mmm_data: MMMData, model_outputs: ModelOutputs):
        """
        Initialize the ImmediateCarryoverCalculator.

        Args:
            mmm_data (MMMData): Input data for the marketing mix model.
            model_outputs (ModelOutputs): Output data from the model runs.
        """
        self.mmm_data = mmm_data
        self.model_outputs = model_outputs

    def calculate(
        self, sol_id: Optional[str] = None, start_date: Optional[str] = None, end_date: Optional[str] = None
    ) -> List[EffectDecomposition]:
        """
        Calculate immediate and carryover effects for media channels.

        Args:
            sol_id (Optional[str]): Solution ID to use. If None, uses the first solution.
            start_date (Optional[str]): Start date for the calculation. If None, uses the window start.
            end_date (Optional[str]): End date for the calculation. If None, uses the window end.

        Returns:
            List[EffectDecomposition]: List of effect decompositions for each channel.
        """
        date_range = self._get_date_range(start_date, end_date)
        sol_id = sol_id or self._get_default_solution_id()

        saturated_dfs = self._calculate_saturated_dataframes(sol_id)
        decomp_data = self._calculate_decomposition(sol_id, saturated_dfs, date_range)

        return self._aggregate_effects(decomp_data, date_range)

    def calculate_all(self) -> pd.DataFrame:
        """
        Calculate immediate and carryover effects for all solutions.

        Returns:
            pd.DataFrame: Dataframe with immediate and carryover effects for all solutions.
        """
        pass

    def _get_date_range(self, start_date: Optional[str], end_date: Optional[str]) -> DateRange:
        """
        Get the date range for the calculation.

        Args:
            start_date (Optional[str]): Start date for the calculation.
            end_date (Optional[str]): End date for the calculation.

        Returns:
            DateRange: Object containing start and end dates and indices.
        """
        pass

    def _calculate_saturated_dataframes(self, sol_id: str) -> Dict[str, pd.DataFrame]:
        """
        Calculate saturated dataframes with carryover and immediate parts.

        Args:
            sol_id (str): Solution ID to use.

        Returns:
            Dict[str, pd.DataFrame]: Dictionary of saturated dataframes.
        """
        pass

    def _calculate_decomposition(
        self, sol_id: str, saturated_dfs: Dict[str, pd.DataFrame], date_range: DateRange
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Calculate decomposition of the effects.

        Args:
            sol_id (str): Solution ID to use.
            saturated_dfs (Dict[str, pd.DataFrame]): Saturated dataframes.
            date_range (DateRange): Date range for the calculation.

        Returns:
            Dict[str, Dict[str, np.ndarray]]: Dictionary of decomposed effects.
                The outer dictionary keys are channel names.
                The inner dictionary has keys 'immediate' and 'carryover',
                each containing an array of effect values for each time point.
        """
        pass

    def _aggregate_effects(
        self, decomp_data: Dict[str, Dict[str, np.ndarray]], date_range: DateRange
    ) -> List[EffectDecomposition]:
        """
        Aggregate decomposed effects into summary statistics for each channel.

        This method takes the raw decomposition data, which contains immediate and carryover
        effects for each time point, and aggregates it into total effects for the entire
        date range for each channel.

        Args:
            decomp_data (Dict[str, Dict[str, np.ndarray]]): Decomposed effects data.
                The outer dictionary keys are channel names.
                The inner dictionary has keys 'immediate' and 'carryover',
                each containing an array of effect values for each time point.
            date_range (DateRange): Date range of the calculation.

        Returns:
            List[EffectDecomposition]: List of aggregated effect decompositions for each channel.
        """
        pass

    def _get_default_solution_id(self) -> str:
        """
        Get the default solution ID (first solution in the model outputs).

        Returns:
            str: Default solution ID.
        """
        pass

    def _get_all_solution_ids(self) -> List[str]:
        """
        Get all solution IDs from the model outputs.

        Returns:
            List[str]: List of all solution IDs.
        """
        pass
