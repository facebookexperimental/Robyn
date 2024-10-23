from typing import List
import pandas as pd
import numpy as np
from robyn.data.entities.mmmdata import MMMData
from robyn.data.entities.hyperparameters import Hyperparameters
from robyn.data.entities.calibration_input import CalibrationInput, ChannelCalibrationData
from robyn.data.entities.enums import AdstockType, CalibrationScope
from robyn.calibration.calibration_transformation import TransformationEngine
from dataclasses import dataclass
from typing import Dict, List
import numpy as np


@dataclass(frozen=True)
class CalibrationResult:
    """
    Immutable data class that holds the results of a calibration run.

    Attributes:
        channel_scores (Dict[str, float]): MAPE scores per channel
        calibration_constraint (float): Acceptable error range (0.01-0.1)
        calibrated_models (List[str]): Model IDs that passed calibration
    """

    channel_scores: Dict[str, float]
    calibration_constraint: float
    calibrated_models: List[str]

    def get_mean_mape(self) -> float:
        """Returns mean absolute percentage error across all channels."""
        return np.mean(list(self.channel_scores.values()))

    def is_model_calibrated(self, mape_threshold: float = 0.1) -> bool:
        """Checks if model meets calibration criteria."""
        return self.get_mean_mape() <= mape_threshold

    def __str__(self) -> str:
        scores_str = "\n".join(f"  {channel}: MAPE = {score:.2%}" for channel, score in self.channel_scores.items())
        return (
            f"CalibrationResult(\n"
            f"  Mean MAPE: {self.get_mean_mape():.2%}\n"
            f"  Constraint: {self.calibration_constraint}\n"
            f"  Channel Scores:\n{scores_str}\n"
            f"  Calibrated: {self.is_model_calibrated()}\n"
            f")"
        )


class CalibrationEngine:
    """
    Handles the calibration process for media mix models by comparing model predictions
    against actual experimental results.
    """

    def __init__(self, mmm_data: MMMData, hyperparameters: Hyperparameters, calibration_input: CalibrationInput):
        """Initialize the calibration engine with model data and parameters."""
        self.mmm_data = mmm_data
        self.hyperparameters = hyperparameters
        self.calibration_input = calibration_input
        self.transformation_engine = TransformationEngine(hyperparameters)

        # Convert date column to datetime if it's not already
        date_col = self.mmm_data.mmmdata_spec.date_var
        if not pd.api.types.is_datetime64_any_dtype(self.mmm_data.data[date_col]):
            self.mmm_data.data[date_col] = pd.to_datetime(self.mmm_data.data[date_col])

        self._validate_inputs()

    def _split_channel(self, channel: str) -> List[str]:
        """Split combined channel names into individual channels."""
        return channel.split("+")

    def _validate_inputs(self) -> None:
        """
        Validates that calibration inputs match model data requirements.
        Handles both single channels and combined channels (joined with '+').
        """
        all_valid_channels = set(
            self.mmm_data.mmmdata_spec.paid_media_spends + self.mmm_data.mmmdata_spec.organic_vars
        )

        for channel in self.calibration_input.channel_data:
            # Split combined channels
            individual_channels = self._split_channel(channel)

            # Check if all individual channels exist in model variables
            invalid_channels = [ch for ch in individual_channels if ch not in all_valid_channels]

            if invalid_channels:
                raise ValueError(
                    f"Channel(s) {', '.join(invalid_channels)} not found in model variables. "
                    f"Available channels: {', '.join(sorted(all_valid_channels))}"
                )

        # Validate dates are within model window
        model_start = pd.to_datetime(self.mmm_data.mmmdata_spec.window_start)
        model_end = pd.to_datetime(self.mmm_data.mmmdata_spec.window_end)

        # Convert model dates to timestamps if they're not already
        if not isinstance(model_start, pd.Timestamp):
            model_start = pd.Timestamp(model_start)
        if not isinstance(model_end, pd.Timestamp):
            model_end = pd.Timestamp(model_end)

        for channel, data in self.calibration_input.channel_data.items():
            lift_start = pd.Timestamp(data.lift_start_date)
            lift_end = pd.Timestamp(data.lift_end_date)

            if not (model_start <= lift_start <= model_end and model_start <= lift_end <= model_end):
                raise ValueError(f"Dates for {channel} outside model window ({model_start} to {model_end})")

    def _get_channel_predictions(self, channel: str, data: ChannelCalibrationData) -> pd.Series:
        """Gets model predictions for a channel during calibration period."""
        date_col = self.mmm_data.mmmdata_spec.date_var

        # Convert dates to timestamps for comparison
        lift_start = pd.Timestamp(data.lift_start_date)
        lift_end = pd.Timestamp(data.lift_end_date)

        mask = (self.mmm_data.data[date_col] >= lift_start) & (self.mmm_data.data[date_col] <= lift_end)
        return self.mmm_data.data.loc[mask, channel]

    def _calculate_immediate_effect_score(
        self, predictions: pd.Series, lift_value: float, spend: float, channel: str
    ) -> float:
        """
        Calculates calibration score for immediate effects.

        Args:
            predictions: Series of predictions to evaluate
            lift_value: Actual lift value to compare against
            spend: Spend amount for the channel
            channel: Channel name for getting correct hyperparameters
        """
        pred_effect = self.transformation_engine.apply_media_transforms(predictions, channel)
        # Sum up the prediction effects for the period
        total_pred_effect = pred_effect.sum()
        # Calculate MAPE using the totals
        mape = np.abs((total_pred_effect - lift_value) / lift_value)
        return float(mape)

    def _calculate_total_effect_score(
        self, predictions: pd.Series, lift_value: float, spend: float, channel: str
    ) -> float:
        """
        Calculates calibration score for total effects including carryover.

        Args:
            predictions: Series of predictions to evaluate
            lift_value: Actual lift value to compare against
            spend: Spend amount for the channel
            channel: Channel name for getting correct hyperparameters
        """
        pred_effect = self.transformation_engine.apply_media_transforms(predictions, channel)
        # Get total effect including carryover
        total_effect = self.transformation_engine.apply_carryover_effect(pred_effect)
        # Calculate MAPE using the totals
        mape = np.abs((total_effect - lift_value) / lift_value)
        return float(mape)

    def calibrate(self) -> CalibrationResult:
        """
        Performs calibration by comparing model predictions to experimental results.
        Returns CalibrationResult with scores per channel.
        """
        calibration_scores = {}

        for channel, data in self.calibration_input.channel_data.items():
            # Handle single or combined channels
            channel_list = self._split_channel(channel)

            try:
                # Get predictions for all relevant channels
                predictions = []
                for ch in channel_list:
                    pred = self._get_channel_predictions(ch, data)
                    predictions.append(pred)

                # Sum predictions for combined channels
                total_pred = pd.concat(predictions, axis=1).sum(axis=1)

                # Use the first channel's hyperparameters for combined channels
                channel_for_params = channel_list[0]

                # Calculate calibration score based on scope
                if data.calibration_scope == CalibrationScope.IMMEDIATE:
                    score = self._calculate_immediate_effect_score(
                        total_pred, data.lift_abs, data.spend, channel_for_params
                    )
                else:
                    score = self._calculate_total_effect_score(
                        total_pred, data.lift_abs, data.spend, channel_for_params
                    )

                calibration_scores[channel] = score

            except Exception as e:
                print(f"Error calculating calibration for channel {channel}: {str(e)}")
                # You might want to handle this differently based on your requirements
                calibration_scores[channel] = float("inf")

        return CalibrationResult(
            channel_scores=calibration_scores,
            calibration_constraint=0.05,  # Default constraint
            calibrated_models=[],  # To be populated with model IDs that pass calibration
        )
