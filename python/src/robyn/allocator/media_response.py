"""
media_response.py - Calculates parameters for media response curves

This module implements the parameter calculation logic from allocator.R's get_hill_params(),
using the ParetoResult data structure from data_mapper.py.
"""

import logging
from dataclasses import dataclass
from typing import Dict
import pandas as pd
import numpy as np

from robyn.data.entities.mmmdata import MMMData
from robyn.modeling.pareto.pareto_optimizer import ParetoResult
from robyn.modeling.pareto.pareto_optimizer import ParetoResult

# Configure logger
logger = logging.getLogger(__name__)

@dataclass
class MediaResponseParameters:
    """Container for media response parameters calculated for each channel."""

    alphas: Dict[str, float]
    inflexions: Dict[str, float]
    coefficients: Dict[str, float]

    def __str__(self) -> str:
        """Custom string representation for logging purposes."""
        return (
            f"MediaResponseParameters(alphas={len(self.alphas)} channels, "
            f"inflexions={len(self.inflexions)} channels, "
            f"coefficients={len(self.coefficients)} channels)"
        )


class MediaResponseParamsCalculator:
    """Calculates response curve parameters for media channels."""

    def __init__(
        self,
        mmm_data: MMMData,
        pareto_result: ParetoResult,
        select_model: str,
    ):
        """Initialize calculator with model data and parameters."""
        logger.info("Initializing MediaResponseParamsCalculator")
        logger.debug("Input parameters: mmm_data=%s, pareto_result=%s, select_model=%s",
                    type(mmm_data).__name__, type(pareto_result).__name__, select_model)
        
        self.mmm_data = mmm_data
        self.pareto_result = pareto_result
        self.select_model = select_model

        # Get media channels in sorted order
        self.media_channels = np.array(self.mmm_data.mmmdata_spec.paid_media_spends)
        self.media_order = np.argsort(self.media_channels)
        self.sorted_channels = self.media_channels[self.media_order]
        
        logger.debug("Initialized with %d media channels: %s", 
                    len(self.sorted_channels), self.sorted_channels.tolist())

    def calculate_parameters(self) -> MediaResponseParameters:
        """Calculate response parameters with improved error handling."""
        logger.info("Starting media response parameters calculation for model %s", self.select_model)
        
        try:
            # Get hyperparameters for selected model
            model_key = "solID" if "solID" in self.pareto_result.result_hyp_param.columns else "sol_id"
            logger.debug("Using model key: %s", model_key)
            
            dt_hyppar = self.pareto_result.result_hyp_param[
                self.pareto_result.result_hyp_param[model_key] == self.select_model
            ]

            if dt_hyppar.empty:
                logger.error("No hyperparameters found for model %s", self.select_model)
                raise ValueError(f"No hyperparameters found for model {self.select_model}")

            # Calculate parameters for each channel
            alphas = {}
            inflexions = {}
            coefficients = {}

            for channel in self.sorted_channels:
                logger.debug("Processing channel: %s", channel)

                # Get alpha parameter
                # Get alpha parameter
                alpha_key = f"{channel}_alphas"
                if alpha_key not in dt_hyppar.columns:
                    logger.warning("Missing %s, using default alpha value of 1.0", alpha_key)
                    alphas[channel] = 1.0
                else:
                    alphas[channel] = dt_hyppar[alpha_key].iloc[0]
                    logger.debug("Channel %s alpha: %f", channel, alphas[channel])

                # Get gamma and calculate inflexion point
                gamma_key = f"{channel}_gammas"
                if gamma_key not in dt_hyppar.columns:
                    logger.warning("Missing %s, using default gamma value of 0.5", gamma_key)
                    gamma = 0.5
                else:
                    gamma = dt_hyppar[gamma_key].iloc[0]
                    logger.debug("Channel %s gamma: %f", channel, gamma)

                # Calculate inflexion
                if channel in self.mmm_data.data.columns:
                    channel_data = self.mmm_data.data[channel]
                    value_range = np.array([channel_data.min(), channel_data.max()])
                    inflexions[channel] = np.dot(value_range, [1 - gamma, gamma])
                    logger.debug("Channel %s inflexion: %f", channel, inflexions[channel])
                else:
                    logger.warning("Channel %s not found in data, using default inflexion", channel)
                    inflexions[channel] = 1.0

                # Get coefficient
                model_key_decomp = "solID" if "solID" in self.pareto_result.x_decomp_agg.columns else "sol_id"
                coef_data = self.pareto_result.x_decomp_agg[
                    (self.pareto_result.x_decomp_agg[model_key_decomp] == self.select_model)
                    & (self.pareto_result.x_decomp_agg["rn"] == channel)
                    (self.pareto_result.x_decomp_agg[model_key_decomp] == self.select_model)
                    & (self.pareto_result.x_decomp_agg["rn"] == channel)
                ]

                if coef_data.empty:
                    logger.warning("No coefficient data found for %s, using default coefficient of 1.0", channel)
                    coefficients[channel] = 1.0
                else:
                    coefficients[channel] = coef_data["coef"].iloc[0]
                    logger.debug("Channel %s coefficient: %f", channel, coefficients[channel])

            result = MediaResponseParameters(alphas=alphas, inflexions=inflexions, coefficients=coefficients)
            logger.info("Successfully calculated media response parameters: %s", result)
            return result

        except Exception as e:
            logger.error("Failed to calculate media response parameters", exc_info=True)
            logger.debug("Debug information - Selected model: %s", self.select_model)
            if hasattr(self.pareto_result, "result_hyp_param"):
                logger.debug("Available columns in result_hyp_param: %s", 
                           self.pareto_result.result_hyp_param.columns.tolist())
            if hasattr(self.pareto_result, "x_decomp_agg"):
                logger.debug("Available columns in x_decomp_agg: %s", 
                           self.pareto_result.x_decomp_agg.columns.tolist())
            raise ValueError(f"Failed to calculate media response parameters: {str(e)}") from e

    def _get_adstocked_media_data(self) -> pd.DataFrame:
        """Get adstocked media data with improved fallback handling."""
        logger.info("Attempting to retrieve adstocked media data")
        
        try:
            # First try to get data from media_vec_collect
            if hasattr(self.pareto_result, "media_vec_collect"):
                media_vec = self.pareto_result.media_vec_collect
                
                if not isinstance(media_vec, pd.DataFrame) or media_vec.empty:
                    logger.warning("media_vec_collect is empty or invalid")
                else:
                    logger.debug("Available columns in media_vec_collect: %s", 
                               media_vec.columns.tolist())

                    # Check if required columns exist
                    model_key = "solID" if "solID" in media_vec.columns else "sol_id"
                    if "type" in media_vec.columns and model_key in media_vec.columns:
                        adstocked_data = media_vec[
                            (media_vec["type"] == "adstockedMedia") & 
                            (media_vec[model_key] == self.select_model)
                        ]

                        if not adstocked_data.empty:
                            logger.info("Successfully retrieved adstocked media data")
                            return adstocked_data[self.sorted_channels]

            # If we reach here, use raw data as fallback
            logger.warning("Using raw media data as fallback")
            raw_data = self.mmm_data.data[self.sorted_channels]

            # Get window indices if available
            start_idx = getattr(self.mmm_data.mmmdata_spec, "rolling_window_start_which", 0)
            end_idx = getattr(self.mmm_data.mmmdata_spec, "rolling_window_end_which", len(raw_data))
            
            logger.debug("Using data window: start_idx=%d, end_idx=%d", start_idx, end_idx)
            return raw_data.iloc[start_idx : end_idx + 1]

        except (KeyError, AttributeError, TypeError) as e:
            logger.error("Error retrieving media data: %s", str(e), exc_info=True)
            logger.warning("Falling back to full raw media data")
            return self.mmm_data.data[self.sorted_channels]