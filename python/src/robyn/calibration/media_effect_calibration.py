# pyre-strict

import logging
from typing import List
import pandas as pd
import numpy as np
from robyn.data.entities.mmmdata import MMMData
from robyn.data.entities.hyperparameters import Hyperparameters
from robyn.data.entities.calibration_input import CalibrationInput, ChannelCalibrationData
from robyn.data.entities.enums import AdstockType, CalibrationScope
from robyn.calibration.media_transformation import MediaTransformation
from dataclasses import dataclass, field
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
    calibration_constraint: float = 0.05  # Default constraint moved to class definition
    calibrated_models: List[str] = field(default_factory=list)  # Empty list default

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


class MediaEffectCalibrator:
    """
    Handles the calibration process for media mix models by comparing model predictions
    against actual experimental results.
    """

    def __init__(self, mmm_data: MMMData, hyperparameters: Hyperparameters, calibration_input: CalibrationInput):
        """Initialize calibration with model data and parameters."""
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing MediaEffectCalibrator")

        self.mmm_data = mmm_data
        self.hyperparameters = hyperparameters
        self.calibration_input = calibration_input
        self.media_transformation = MediaTransformation(hyperparameters)

        # Convert date column to datetime if it's not already
        date_col = self.mmm_data.mmmdata_spec.date_var
        if not pd.api.types.is_datetime64_any_dtype(self.mmm_data.data[date_col]):
            self.mmm_data.data[date_col] = pd.to_datetime(self.mmm_data.data[date_col])

        self._validate_inputs()

    def _validate_inputs(self) -> None:
        """
        Validates that calibration inputs match model data requirements.
        Validates both individual channels and channel combinations.
        """
        all_valid_channels = set(
            self.mmm_data.mmmdata_spec.paid_media_spends + self.mmm_data.mmmdata_spec.organic_vars
        )

        for channel_key in self.calibration_input.channel_data:
            # Convert tuple to list of channels
            if isinstance(channel_key, tuple):
                channels = list(channel_key)
            else:
                channels = [channel_key]

            # Check if all individual channels exist in model variables
            invalid_channels = [ch for ch in channels if ch not in all_valid_channels]

            if invalid_channels:
                error_msg = (
                    f"Channel(s) {', '.join(invalid_channels)} not found in model variables. "
                    f"Available channels: {', '.join(sorted(all_valid_channels))}"
                )
                self.logger.error(error_msg)
                raise ValueError(error_msg)
        # Validate dates are within model window
        model_start = pd.to_datetime(self.mmm_data.mmmdata_spec.window_start)
        model_end = pd.to_datetime(self.mmm_data.mmmdata_spec.window_end)

        # Convert model dates to timestamps if they're not already
        if not isinstance(model_start, pd.Timestamp):
            model_start = pd.Timestamp(model_start)
        if not isinstance(model_end, pd.Timestamp):
            model_end = pd.Timestamp(model_end)

        for channel_key, data in self.calibration_input.channel_data.items():
            lift_start = pd.Timestamp(data.lift_start_date)
            lift_end = pd.Timestamp(data.lift_end_date)

            if not (model_start <= lift_start <= model_end and model_start <= lift_end <= model_end):
                error_msg = f"Dates for {channel_key} outside model window ({model_start} to {model_end})"
                self.logger.error(error_msg)
                raise ValueError(error_msg)

    def _get_channel_predictions(self, channel_key: str, data: ChannelCalibrationData) -> pd.Series:
        """
        Gets model predictions for a channel or combination of channels during calibration period.
        """
        date_col = self.mmm_data.mmmdata_spec.date_var
        lift_start = pd.Timestamp(data.lift_start_date)
        lift_end = pd.Timestamp(data.lift_end_date)

        mask = (self.mmm_data.data[date_col] >= lift_start) & (self.mmm_data.data[date_col] <= lift_end)

        # Handle channel_key whether it's a tuple or string
        if isinstance(channel_key, tuple):
            channels = list(channel_key)
        else:
            channels = [channel_key]

        # Initialize predictions with zeros
        predictions = pd.Series(0, index=self.mmm_data.data.loc[mask].index)

        # Sum predictions for all channels
        for channel in channels:
            predictions += self.mmm_data.data.loc[mask, channel]

        return predictions

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
        pred_effect = self.media_transformation.apply_media_transforms(predictions, channel)
        total_pred_effect = pred_effect.sum()
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
        pred_effect = self.media_transformation.apply_media_transforms(predictions, channel)
        total_effect = self.media_transformation.apply_carryover_effect(pred_effect)
        mape = np.abs((total_effect - lift_value) / lift_value)
        return float(mape)

    def calibrate(self) -> CalibrationResult:
        """
        Performs calibration by comparing model predictions to experimental results.
        Returns CalibrationResult with scores per channel.
        """
        self.logger.info("Starting calibration process")
        calibration_scores = {}

        for channels, data in self.calibration_input.channel_data.items():
            try:
                # Get predictions for channel combination
                predictions = self._get_channel_predictions(channels, data)

                # Use the first channel's parameters for transformations
                channel_for_params = channels[0]

                # Calculate calibration score based on scope
                if data.calibration_scope == CalibrationScope.IMMEDIATE:
                    score = self._calculate_immediate_effect_score(
                        predictions, data.lift_abs, data.spend, channel_for_params
                    )
                else:
                    score = self._calculate_total_effect_score(
                        predictions, data.lift_abs, data.spend, channel_for_params
                    )

                # Convert channels list to string key for backwards compatibility
                channel_key = "+".join(channels)
                calibration_scores[channel_key] = score

            except Exception as e:
                channel_key = "+".join(channels)
                error_msg = f"Error calculating calibration for {channel_key}: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                calibration_scores[channel_key] = float("inf")

        result = CalibrationResult(channel_scores=calibration_scores)
        self.logger.info(f"Calibration complete. Mean MAPE: {result.get_mean_mape():.4f}")
        return result
