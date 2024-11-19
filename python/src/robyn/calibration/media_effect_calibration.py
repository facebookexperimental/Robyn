# pyre-strict

import logging
from typing import List
import pandas as pd
from robyn.data.entities.mmmdata import MMMData
from robyn.data.entities.hyperparameters import Hyperparameters
from robyn.data.entities.calibration_input import (
    CalibrationInput,
    ChannelCalibrationData,
)
from robyn.data.entities.enums import CalibrationScope
from robyn.calibration.media_transformation import MediaTransformation
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional
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
        mean_mape = self.get_mean_mape()
        return mean_mape <= mape_threshold

    def __str__(self) -> str:
        scores_str = "\n".join(
            f"  {channel}: MAPE = {score:.2%}"
            for channel, score in self.channel_scores.items()
        )
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
        model_coefficients: Optional[Dict[str, float]] = None,
    ):
        """Initialize calibration with model data and parameters."""
        self.logger = logging.getLogger(__name__)

        self.logger.info("Initializing MediaEffectCalibrator")
        self.logger.debug(
            "Input parameters: MMMData shape: %s, Hyperparameters count: %d, Calibration channels: %s",
            mmm_data.data.shape if mmm_data.data is not None else "None",
            len(hyperparameters.channel_hyperparameters) if hyperparameters else 0,
            (
                list(calibration_input.channel_data.keys())
                if calibration_input
                else "None"
            ),
        )

        self.mmm_data = mmm_data
        self.hyperparameters = hyperparameters
        self.calibration_input = calibration_input
        self.media_transformation = MediaTransformation(hyperparameters)
        self.model_coefficients = model_coefficients or {}

        # Convert date column to datetime if it's not already
        date_col = self.mmm_data.mmmdata_spec.date_var
        if not pd.api.types.is_datetime64_any_dtype(self.mmm_data.data[date_col]):
            self.logger.debug("Converting date column '%s' to datetime", date_col)
            self.mmm_data.data[date_col] = pd.to_datetime(self.mmm_data.data[date_col])

        self._validate_inputs()
        self.logger.info("MediaEffectCalibrator initialization complete")

    def _validate_inputs(self) -> None:
        """Validates that calibration inputs match model data requirements."""
        self.logger.debug("Starting input validation")

        all_valid_channels = set(
            self.mmm_data.mmmdata_spec.paid_media_spends
            + self.mmm_data.mmmdata_spec.organic_vars
        )
        self.logger.debug("Valid channels: %s", sorted(all_valid_channels))

        for channel_key in self.calibration_input.channel_data:
            if isinstance(channel_key, tuple):
                channels = list(channel_key)
            else:
                channels = [channel_key]

            self.logger.debug("Validating channel(s): %s", channels)
            invalid_channels = [ch for ch in channels if ch not in all_valid_channels]

            if invalid_channels:
                error_msg = (
                    f"Channel(s) {', '.join(invalid_channels)} not found in model variables. "
                    f"Available channels: {', '.join(sorted(all_valid_channels))}"
                )
                self.logger.error(error_msg)
                raise ValueError(error_msg)

        # Validate dates
        model_start = pd.to_datetime(self.mmm_data.mmmdata_spec.window_start)
        model_end = pd.to_datetime(self.mmm_data.mmmdata_spec.window_end)

        self.logger.debug("Model window: %s to %s", model_start, model_end)

        for channel_key, data in self.calibration_input.channel_data.items():
            lift_start = pd.Timestamp(data.lift_start_date)
            lift_end = pd.Timestamp(data.lift_end_date)

            self.logger.debug(
                "Validating dates for %s: %s to %s", channel_key, lift_start, lift_end
            )

            if not (
                model_start <= lift_start <= model_end
                and model_start <= lift_end <= model_end
            ):
                error_msg = f"Dates for {channel_key} outside model window ({model_start} to {model_end})"
                self.logger.error(error_msg)
                raise ValueError(error_msg)

        self.logger.info("Input validation completed successfully")

    def calculate_immediate_effect_score(
        self,
        predictions: pd.Series,
        lift_value: float,
        spend: float,
        channel: str,
        data: ChannelCalibrationData,
    ) -> float:
        """Calculates immediate effect score with detailed logging."""
        self.logger.debug("Calculating immediate effect for channel: %s", channel)
        self.logger.debug(
            "Input parameters - Lift value: %.2f, Spend: %.2f", lift_value, spend
        )

        # Get channel parameters
        channel_params = self.hyperparameters.get_hyperparameter(channel)
        theta = channel_params.thetas[0]
        alpha = channel_params.alphas[0]
        gamma = channel_params.gammas[0]
        coef = self.model_coefficients.get(channel, 1.0)

        self.logger.debug(
            "Channel parameters - Theta: %.3f, Alpha: %.3f, Gamma: %.3f, Coefficient: %.3f",
            theta,
            alpha,
            gamma,
            coef,
        )

        # Calculate effects
        m_imme = predictions.copy()
        m_total = self.media_transformation._apply_geometric_adstock(predictions, theta)
        m_caov = m_total - m_imme
        m_caov_sat = self.media_transformation._apply_saturation(m_caov, alpha, gamma)
        m_caov_decomp = m_caov_sat * coef

        lift_days = (
            pd.Timestamp(data.lift_end_date) - pd.Timestamp(data.lift_start_date)
        ).days + 1
        decomp_days = len(predictions)
        scaled_effect = (m_caov_decomp.sum() / decomp_days) * lift_days

        self.logger.debug(
            "Effect calculations - Raw sum: %.2f, Total adstock: %.2f, Carryover: %.2f",
            predictions.sum(),
            m_total.sum(),
            m_caov.sum(),
        )
        self.logger.debug(
            "Saturated carryover: %.2f, Final scaled effect: %.2f, Target lift: %.2f",
            m_caov_sat.sum(),
            scaled_effect,
            lift_value,
        )

        mape = np.abs((scaled_effect - lift_value) / lift_value)
        self.logger.info("Channel %s immediate effect MAPE: %.4f", channel, mape)

        return float(mape)

    def calculate_total_effect_score(
        self,
        predictions: pd.Series,
        lift_value: float,
        spend: float,
        channel: str,
        data: ChannelCalibrationData,
    ) -> float:
        """Calculates total effect score with detailed logging."""
        self.logger.debug("Calculating total effect for channel: %s", channel)
        self.logger.debug(
            "Input parameters - Lift value: %.2f, Spend: %.2f", lift_value, spend
        )

        # Get channel parameters
        channel_params = self.hyperparameters.get_hyperparameter(channel)
        theta = channel_params.thetas[0]
        alpha = channel_params.alphas[0]
        gamma = channel_params.gammas[0]
        coef = self.model_coefficients.get(channel, 1.0)

        self.logger.debug(
            "Channel parameters - Theta: %.3f, Alpha: %.3f, Gamma: %.3f, Coefficient: %.3f",
            theta,
            alpha,
            gamma,
            coef,
        )

        # Calculate effects
        m_imme = predictions.copy()
        m_total = self.media_transformation._apply_geometric_adstock(predictions, theta)
        m_caov = m_total - m_imme

        # Calculate scaling factor
        if spend > 0:
            revenue_per_spend = lift_value / spend
            scale_factor = revenue_per_spend
            self.logger.debug("Using spend-based scaling factor: %.4f", scale_factor)
        else:
            mean_value = predictions.mean()
            scale_factor = lift_value / (mean_value if mean_value > 0 else 1.0)
            self.logger.debug("Using mean-value scaling factor: %.4f", scale_factor)

        m_caov_sat = self.media_transformation._apply_saturation(
            m_caov * scale_factor, alpha, gamma
        )
        m_caov_decomp = m_caov_sat * coef

        lift_days = (
            pd.Timestamp(data.lift_end_date) - pd.Timestamp(data.lift_start_date)
        ).days + 1
        decomp_days = len(predictions)
        scaled_effect = (m_caov_decomp.sum() / decomp_days) * lift_days

        self.logger.debug(
            "Effect calculations - Raw sum: %.2f, Total adstock: %.2f, Carryover: %.2f",
            predictions.sum(),
            m_total.sum(),
            m_caov.sum(),
        )
        self.logger.debug(
            "Saturated carryover: %.2f, Final scaled effect: %.2f, Target lift: %.2f",
            m_caov_sat.sum(),
            scaled_effect,
            lift_value,
        )

        mape = np.abs((scaled_effect - lift_value) / lift_value)
        self.logger.info("Channel %s total effect MAPE: %.4f", channel, mape)

        return float(mape)

    def _get_channel_predictions(
        self, channel_tuple: Tuple[str, ...], data: ChannelCalibrationData
    ) -> pd.Series:
        """Gets model predictions for channels during calibration period."""
        self.logger.debug("Getting predictions for channels: %s", channel_tuple)

        date_col = self.mmm_data.mmmdata_spec.date_var
        lift_start = pd.Timestamp(data.lift_start_date)
        lift_end = pd.Timestamp(data.lift_end_date)
        dates = self.mmm_data.data[date_col]

        self.logger.debug("Prediction period: %s to %s", lift_start, lift_end)

        if lift_start in dates.values:
            mask = (dates >= lift_start) & (dates <= lift_end)
        else:
            first_valid = dates[dates > lift_start].min()
            self.logger.warning(
                "Lift start date %s not found, using first valid date %s",
                lift_start,
                first_valid,
            )
            mask = (dates >= (first_valid - pd.Timedelta(days=1))) & (dates <= lift_end)

        predictions = pd.Series(0, index=self.mmm_data.data.loc[mask].index)

        for channel in channel_tuple:
            channel_values = self.mmm_data.data.loc[mask, channel]
            self.logger.debug(
                "Channel %s values - Mean: %.2f, Sum: %.2f",
                channel,
                channel_values.mean(),
                channel_values.sum(),
            )

            if channel in self.mmm_data.mmmdata_spec.paid_media_spends:
                predictions += channel_values
            else:
                predictions += channel_values

        self.logger.debug(
            "Final predictions - Mean: %.2f, Sum: %.2f",
            predictions.mean(),
            predictions.sum(),
        )
        return predictions

    def calibrate(self) -> CalibrationResult:
        """Performs calibration with comprehensive logging."""
        self.logger.info("Starting calibration process")
        calibration_scores = {}

        for channel_tuple, data in self.calibration_input.channel_data.items():
            self.logger.info("Processing channel(s): %s", channel_tuple)
            try:
                predictions = self._get_channel_predictions(channel_tuple, data)
                channel_for_params = channel_tuple[0]

                self.logger.debug("Calibration scope: %s", data.calibration_scope)
                if data.calibration_scope == CalibrationScope.IMMEDIATE:
                    score = self.calculate_immediate_effect_score(
                        predictions, data.lift_abs, data.spend, channel_for_params, data
                    )
                else:
                    score = self.calculate_total_effect_score(
                        predictions, data.lift_abs, data.spend, channel_for_params, data
                    )

                calibration_scores[channel_tuple] = score
                self.logger.info(
                    "Channel %s calibration score: %.4f", channel_tuple, score
                )

            except Exception as e:
                error_msg = (
                    f"Error calculating calibration for {channel_tuple}: {str(e)}"
                )
                self.logger.error(error_msg, exc_info=True)
                calibration_scores[channel_tuple] = float("inf")

        result = CalibrationResult(channel_scores=calibration_scores)
        self.logger.info("Calibration complete - Final results: %s", str(result))
        return result
