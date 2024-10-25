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
from typing import Dict, List, Tuple, Optional
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

    def __init__(
        self,
        mmm_data: MMMData,
        hyperparameters: Hyperparameters,
        calibration_input: CalibrationInput,
        model_coefficients: Optional[Dict[str, float]] = None,  # Add this parameter
    ):
        """Initialize calibration with model data and parameters."""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)  # Set to DEBUG for more detailed output
        self.logger.info("Initializing MediaEffectCalibrator")

        self.mmm_data = mmm_data
        self.hyperparameters = hyperparameters
        self.calibration_input = calibration_input
        self.media_transformation = MediaTransformation(hyperparameters)
        self.model_coefficients = model_coefficients or {}

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

    def calculate_immediate_effect_score(
        self, predictions: pd.Series, lift_value: float, spend: float, channel: str, data: ChannelCalibrationData
    ) -> float:
        """
        Calculates immediate effect score with debugging.
        """
        self.logger.debug(f"\nCalculating immediate effect for {channel}")

        # Get channel parameters
        channel_params = self.hyperparameters.get_hyperparameter(channel)
        theta = channel_params.thetas[0]
        alpha = channel_params.alphas[0]
        gamma = channel_params.gammas[0]
        coef = self.model_coefficients.get(channel, 1.0)

        # 1. Adstock
        m_imme = predictions.copy()  # immediate effect is raw value
        m_total = self.media_transformation._apply_geometric_adstock(predictions, theta)
        m_caov = m_total - m_imme  # carryover effect

        # 2. Apply saturation only to carryover component
        m_caov_sat = self.media_transformation._apply_saturation(m_caov, alpha, gamma)

        # 3. Apply coefficient to saturated carryover
        m_caov_decomp = m_caov_sat * coef

        # 4. Scale effect based on time window
        lift_days = (pd.Timestamp(data.lift_end_date) - pd.Timestamp(data.lift_start_date)).days + 1
        decomp_days = len(predictions)
        scaled_effect = (m_caov_decomp.sum() / decomp_days) * lift_days

        self.logger.debug(f"Steps for {channel}:")
        self.logger.debug(f"  Raw values sum: {predictions.sum():,.2f}")
        self.logger.debug(f"  Adstock total: {m_total.sum():,.2f}")
        self.logger.debug(f"  Carryover: {m_caov.sum():,.2f}")
        self.logger.debug(f"  Saturated carryover: {m_caov_sat.sum():,.2f}")
        self.logger.debug(f"  After coefficient: {m_caov_decomp.sum():,.2f}")
        self.logger.debug(f"  Final scaled: {scaled_effect:,.2f}")
        self.logger.debug(f"  Target lift: {lift_value:,.2f}")

        # Calculate MAPE
        mape = np.abs((scaled_effect - lift_value) / lift_value)
        self.logger.debug(f"MAPE: {mape:.4%}")

        return float(mape)

    def calculate_total_effect_score(
        self, predictions: pd.Series, lift_value: float, spend: float, channel: str, data: ChannelCalibrationData
    ) -> float:
        """
        Calculates total effect score including carryover.
        """
        # Get channel parameters
        channel_params = self.hyperparameters.get_hyperparameter(channel)
        theta = channel_params.thetas[0]
        alpha = channel_params.alphas[0]
        gamma = channel_params.gammas[0]
        coef = self.model_coefficients.get(channel, 1.0)

        # 1. Adstock
        m_imme = predictions.copy()
        m_total = self.media_transformation._apply_geometric_adstock(predictions, theta)
        m_caov = m_total - m_imme

        # Calculate revenue per unit spend for scaling
        if spend > 0:
            revenue_per_spend = lift_value / spend
            scale_factor = revenue_per_spend
        else:
            # For organic channels, use mean value scaling
            mean_value = predictions.mean()
            scale_factor = lift_value / (mean_value if mean_value > 0 else 1.0)

        # 2. Apply saturation and scale the carryover
        m_caov_sat = self.media_transformation._apply_saturation(m_caov * scale_factor, alpha, gamma)

        # 3. Apply coefficient
        m_caov_decomp = m_caov_sat * coef

        # 4. Time window scaling
        lift_days = (pd.Timestamp(data.lift_end_date) - pd.Timestamp(data.lift_start_date)).days + 1
        decomp_days = len(predictions)
        scaled_effect = (m_caov_decomp.sum() / decomp_days) * lift_days

        self.logger.debug(f"Steps for {channel}:")
        self.logger.debug(f"  Raw values sum: {predictions.sum():,.2f}")
        self.logger.debug(f"  Scale factor: {scale_factor:,.4f}")
        self.logger.debug(f"  Adstock total: {m_total.sum():,.2f}")
        self.logger.debug(f"  Carryover: {m_caov.sum():,.2f}")
        self.logger.debug(f"  Saturated carryover: {m_caov_sat.sum():,.2f}")
        self.logger.debug(f"  After coefficient: {m_caov_decomp.sum():,.2f}")
        self.logger.debug(f"  Final scaled: {scaled_effect:,.2f}")
        self.logger.debug(f"  Target lift: {lift_value:,.2f}")

        # Calculate MAPE
        mape = np.abs((scaled_effect - lift_value) / lift_value)
        self.logger.debug(f"MAPE: {mape:.4%}")

        return float(mape)

    def _get_channel_predictions(self, channel_tuple: Tuple[str, ...], data: ChannelCalibrationData) -> pd.Series:
        """Gets model predictions for channels during calibration period with proper scaling."""
        date_col = self.mmm_data.mmmdata_spec.date_var
        lift_start = pd.Timestamp(data.lift_start_date)
        lift_end = pd.Timestamp(data.lift_end_date)
        dates = self.mmm_data.data[date_col]
        if lift_start in dates.values:
            mask = (dates >= lift_start) & (dates <= lift_end)
        else:
            first_valid = dates[dates > lift_start].min()
            mask = (dates >= (first_valid - pd.Timedelta(days=1))) & (dates <= lift_end)

        predictions = pd.Series(0, index=self.mmm_data.data.loc[mask].index)

        # Sum predictions with proper scaling
        for channel in channel_tuple:
            channel_values = self.mmm_data.data.loc[mask, channel]

            # Scale values if needed based on channel type
            if channel in self.mmm_data.mmmdata_spec.paid_media_spends:
                predictions += channel_values
            else:
                # For organic channels, use different scaling
                predictions += channel_values

        return predictions

    def calibrate(self) -> CalibrationResult:
        """
        Performs calibration by comparing model predictions to experimental results.
        Returns CalibrationResult with scores per channel.
        """
        self.logger.info("Starting calibration process")
        calibration_scores = {}

        for channel_tuple, data in self.calibration_input.channel_data.items():
            try:
                # Get predictions for channel combination
                predictions = self._get_channel_predictions(channel_tuple, data)
                channel_for_params = channel_tuple[0]

                if data.calibration_scope == CalibrationScope.IMMEDIATE:
                    score = self.calculate_immediate_effect_score(
                        predictions, data.lift_abs, data.spend, channel_for_params, data
                    )
                else:
                    score = self.calculate_total_effect_score(
                        predictions, data.lift_abs, data.spend, channel_for_params, data
                    )

                calibration_scores[channel_tuple] = score

            except Exception as e:
                error_msg = f"Error calculating calibration for {channel_tuple}: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                calibration_scores[channel_tuple] = float("inf")

        result = CalibrationResult(channel_scores=calibration_scores)
        self.logger.info(f"Calibration complete. Mean MAPE: {result.get_mean_mape():.4f}")
        return result
