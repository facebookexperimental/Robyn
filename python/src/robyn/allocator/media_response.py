"""
media_response.py - Calculates parameters for media response curves

This module implements the parameter calculation logic from allocator.R's get_hill_params(),
using the ParetoResult data structure from data_mapper.py.
"""

from dataclasses import dataclass
from typing import Dict, Optional
import pandas as pd
import numpy as np

from robyn.data.entities.mmmdata import MMMData
from robyn.modeling.pareto.pareto_optimizer import ParetoResult


@dataclass
class MediaResponseParameters:
    """Container for media response parameters calculated for each channel."""

    alphas: Dict[str, float]
    inflexions: Dict[str, float]
    coefficients: Dict[str, float]


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
        self.mmm_data = mmm_data
        self.pareto_result = pareto_result
        self.select_model = select_model

        # Get media channels in sorted order
        self.media_channels = np.array(self.mmm_data.mmmdata_spec.paid_media_spends)
        self.media_order = np.argsort(self.media_channels)
        self.sorted_channels = self.media_channels[self.media_order]

    def calculate_parameters(self) -> MediaResponseParameters:
        """Calculate response parameters for all media channels."""
        # Get hyperparameters for selected model
        dt_hyppar = self.pareto_result.result_hyp_param[
            self.pareto_result.result_hyp_param["solID"] == self.select_model
        ]

        # Get adstocked media data
        adstocked_data = self._get_adstocked_media_data()

        # Calculate parameters for each channel
        alphas = {}
        inflexions = {}
        coefficients = {}

        for channel in self.sorted_channels:
            # Get alpha parameter
            alpha_key = f"{channel}_alphas"
            alphas[channel] = dt_hyppar[alpha_key].iloc[0]

            # Get gamma and calculate inflexion point
            gamma_key = f"{channel}_gammas"
            gamma = dt_hyppar[gamma_key].iloc[0]

            channel_data = adstocked_data[channel]
            value_range = np.array([channel_data.min(), channel_data.max()])
            inflexions[channel] = np.dot(value_range, [1 - gamma, gamma])

            # Get coefficient
            coef_data = self.pareto_result.x_decomp_agg[
                (self.pareto_result.x_decomp_agg["solID"] == self.select_model)
                & (self.pareto_result.x_decomp_agg["rn"] == channel)
            ]
            coefficients[channel] = coef_data["coef"].iloc[0]

        return MediaResponseParameters(alphas=alphas, inflexions=inflexions, coefficients=coefficients)

    def _get_adstocked_media_data(self) -> pd.DataFrame:
        """Get adstocked media data for the selected model within the specified window."""
        # Filter for adstocked media data
        adstocked_data = self.pareto_result.media_vec_collect[
            (self.pareto_result.media_vec_collect["type"] == "adstockedMedia")
            & (self.pareto_result.media_vec_collect["solID"] == self.select_model)
        ][self.sorted_channels]

        # Get window indices
        start_idx = self.mmm_data.mmmdata_spec.rolling_window_start_which
        end_idx = self.mmm_data.mmmdata_spec.rolling_window_end_which

        # Return windowed data
        return adstocked_data.iloc[start_idx : end_idx + 1]

    def get_parameter_summary(self) -> pd.DataFrame:
        """Generate a summary DataFrame of all parameters."""
        params = self.calculate_parameters()

        return pd.DataFrame(
            {
                "channel": list(self.sorted_channels),
                "alpha": [params.alphas[ch] for ch in self.sorted_channels],
                "inflexion": [params.inflexions[ch] for ch in self.sorted_channels],
                "coefficient": [params.coefficients[ch] for ch in self.sorted_channels],
            }
        )
