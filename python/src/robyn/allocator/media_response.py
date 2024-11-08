"""
media_response.py - Calculates parameters for media response curves with enhanced logging.

This module implements the parameter calculation logic from allocator.R's get_hill_params(),
using the ParetoResult data structure from data_mapper.py.
"""

from dataclasses import dataclass
from typing import Dict, Optional
import pandas as pd
import numpy as np
import logging

from robyn.data.entities.mmmdata import MMMData
from robyn.modeling.pareto.pareto_optimizer import ParetoResult
from robyn.modeling.pareto.pareto_optimizer import ParetoResult


@dataclass
class MediaResponseParameters:
    """Container for media response parameters calculated for each channel."""
    alphas: Dict[str, float]
    inflexions: Dict[str, float]
    coefficients: Dict[str, float]

    def __str__(self) -> str:
        """String representation for logging purposes"""
        return (f"MediaResponseParameters(\n"
                f"  alphas: {self.alphas},\n"
                f"  inflexions: {self.inflexions},\n"
                f"  coefficients: {self.coefficients}\n)")


class MediaResponseParamsCalculator:
    """Calculates response curve parameters for media channels."""

    def __init__(
        self,
        mmm_data: MMMData,
        pareto_result: ParetoResult,
        select_model: str,
    ):
        """Initialize calculator with model data and parameters.

        Args:
            mmm_data: Marketing mix modeling data
            pareto_result: Pareto optimization results containing model data
            select_model: Selected model identifier
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing MediaResponseParamsCalculator")
        self.logger.debug(f"Using model ID: {select_model}")

        self.mmm_data = mmm_data
        self.pareto_result = pareto_result
        self.select_model = select_model

        # Get media channels in sorted order
        self.media_channels = np.array(self.mmm_data.mmmdata_spec.paid_media_spends)
        self.media_order = np.argsort(self.media_channels)
        self.sorted_channels = self.media_channels[self.media_order]

        self.logger.debug(f"Initialized with {len(self.sorted_channels)} media channels")
        self.logger.debug(f"Media channels: {', '.join(self.sorted_channels)}")

    def calculate_parameters(self) -> MediaResponseParameters:
        """Calculate response parameters with improved error handling."""
        try:
            # Get hyperparameters for selected model
            model_key = "solID" if "solID" in self.pareto_result.result_hyp_param.columns else "sol_id"
            dt_hyppar = self.pareto_result.result_hyp_param[
                self.pareto_result.result_hyp_param[model_key] == self.select_model
            ]

            if dt_hyppar.empty:
                raise ValueError(f"No hyperparameters found for model {self.select_model}")

            print("\nCalculating media response parameters...")
            print(f"Using model ID: {self.select_model}")

            # Calculate parameters for each channel
            alphas = {}
            inflexions = {}
            coefficients = {}

            for channel in self.sorted_channels:
                print(f"\nProcessing channel: {channel}")

                # Get alpha parameter
                alpha_key = f"{channel}_alphas"
                if alpha_key not in dt_hyppar.columns:
                    print(f"Warning: Missing {alpha_key}, using default alpha value of 1.0")
                    alphas[channel] = 1.0
                else:
                    alphas[channel] = dt_hyppar[alpha_key].iloc[0]
                    print(f"Alpha: {alphas[channel]}")

                # Get gamma and calculate inflexion point
                gamma_key = f"{channel}_gammas"
                if gamma_key not in dt_hyppar.columns:
                    print(f"Warning: Missing {gamma_key}, using default gamma value of 0.5")
                    gamma = 0.5
                else:
                    gamma = dt_hyppar[gamma_key].iloc[0]
                    print(f"Gamma: {gamma}")

                # Calculate inflexion
                if channel in self.mmm_data.data.columns:
                    channel_data = self.mmm_data.data[channel]
                    value_range = np.array([channel_data.min(), channel_data.max()])
                    inflexions[channel] = np.dot(value_range, [1 - gamma, gamma])
                    print(f"Inflexion: {inflexions[channel]}")
                else:
                    print(f"Warning: Channel {channel} not found in data, using default inflexion")
                    inflexions[channel] = 1.0

                # Get coefficient
                model_key_decomp = "solID" if "solID" in self.pareto_result.x_decomp_agg.columns else "sol_id"
                coef_data = self.pareto_result.x_decomp_agg[
                    (self.pareto_result.x_decomp_agg[model_key_decomp] == self.select_model)
                    & (self.pareto_result.x_decomp_agg["rn"] == channel)
                ]

                if coef_data.empty:
                    print(f"Warning: No coefficient data found for {channel}, using default coefficient of 1.0")
                    coefficients[channel] = 1.0
                else:
                    coefficients[channel] = coef_data["coef"].iloc[0]
                    print(f"Coefficient: {coefficients[channel]}")

            return MediaResponseParameters(alphas=alphas, inflexions=inflexions, coefficients=coefficients)

        except Exception as e:
            print("\nError calculating media response parameters:")
            print(f"Selected model: {self.select_model}")
            if hasattr(self.pareto_result, "result_hyp_param"):
                print(f"Available columns in result_hyp_param: {self.pareto_result.result_hyp_param.columns.tolist()}")
            if hasattr(self.pareto_result, "x_decomp_agg"):
                print(f"Available columns in x_decomp_agg: {self.pareto_result.x_decomp_agg.columns.tolist()}")
            raise ValueError(f"Failed to calculate media response parameters: {str(e)}")

    def _get_adstocked_media_data(self) -> pd.DataFrame:
        """Get adstocked media data with improved fallback handling."""
        print("\nAttempting to get adstocked media data...")

        try:
            # First try to get data from media_vec_collect
            if hasattr(self.pareto_result, "media_vec_collect"):
                media_vec = self.pareto_result.media_vec_collect

                if not isinstance(media_vec, pd.DataFrame) or media_vec.empty:
                    print("Warning: media_vec_collect is empty or invalid")
                else:
                    print(f"Available columns in media_vec_collect: {media_vec.columns.tolist()}")

                    # Check if required columns exist
                    model_key = "solID" if "solID" in media_vec.columns else "sol_id"
                    if "type" in media_vec.columns and model_key in media_vec.columns:
                        adstocked_data = media_vec[
                            (media_vec["type"] == "adstockedMedia") & (media_vec[model_key] == self.select_model)
                        ]

                        if not adstocked_data.empty:
                            print("Successfully found adstocked media data")
                            return adstocked_data[self.sorted_channels]

            # If we reach here, use raw data as fallback
            print("Using raw media data as fallback")
            raw_data = self.mmm_data.data[self.sorted_channels]

            # Get window indices if available
            start_idx = getattr(self.mmm_data.mmmdata_spec, "rolling_window_start_which", 0)
            end_idx = getattr(self.mmm_data.mmmdata_spec, "rolling_window_end_which", len(raw_data))

            return raw_data.iloc[start_idx : end_idx + 1]

        except Exception as e:
            print(f"Error getting media data: {str(e)}")
            print("Falling back to full raw media data")
            return self.mmm_data.data[self.sorted_channels]
