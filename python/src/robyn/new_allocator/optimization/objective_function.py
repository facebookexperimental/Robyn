# pyre-strict

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ObjectiveFunction:
    """Handles response curve calculations and optimization objective functions."""

    def __init__(
        self,
        coef_dict: Dict[str, float],
        alphas_dict: Dict[str, float],
        gammas_dict: Dict[str, float],
        hist_carryover_dict: Dict[str, np.ndarray],
    ) -> None:
        """Initialize objective function calculator."""
        self.coef_dict = coef_dict
        self.alphas_dict = alphas_dict
        self.gammas_dict = gammas_dict
        self.hist_carryover_dict = hist_carryover_dict
        logger.debug("Initialized ObjectiveFunction with %d channels", len(coef_dict))

    def calculate_response(self, x: np.ndarray, channel_name: str, get_sum: bool = True) -> float:
        """Calculates response for given spend level."""
        print(f"\nCalculating response for {channel_name}")
        print(f"Input spend: {x}")

        # Get parameters
        coef = self.coef_dict[channel_name]
        alpha = self.alphas_dict[f"{channel_name}_alphas"]
        gamma = self.gammas_dict[f"{channel_name}_gammas"]

        print(f"Parameters: coef={coef}, alpha={alpha}, gamma={gamma}")

        # Add historical carryover effect
        x_hist_carryover = np.mean(self.hist_carryover_dict[channel_name])
        x_adstocked = x + x_hist_carryover

        # Hill transformation exactly matching R implementation
        hill_term = (gamma / x_adstocked) ** alpha
        response = coef / (1 + hill_term)

        print(f"Response: {response}")
        return float(np.sum(response)) if get_sum else response

    def calculate_gradient(
        self,
        x: np.ndarray,
        channel_name: str,
        x_hist_carryover: Optional[float] = None,
    ) -> float:
        """Calculates gradient of response curve at given spend level."""
        coef = self.coef_dict[channel_name]
        alpha = self.alphas_dict[f"{channel_name}_alphas"]
        gamma = self.gammas_dict[f"{channel_name}_gammas"]

        if x_hist_carryover is None:
            x_hist_carryover = np.mean(self.hist_carryover_dict[channel_name])

        x_adstocked = x + x_hist_carryover
        x_scaled = x_adstocked / gamma

        # Modified gradient calculation with better numerical stability
        gradient = coef * alpha * x_scaled ** (alpha - 1) / (gamma * (1 + x_scaled**alpha) ** 2)

        return float(np.sum(gradient))

    def calculate_marginal_response(
        self,
        x: float,
        channel_name: str,
        spend_delta: float = 1.0,
    ) -> float:
        """Calculates marginal response for additional spend.

        Args:
            x: Current spend level
            channel_name: Name of channel
            spend_delta: Additional spend amount

        Returns:
            Marginal response value
        """
        response_current = self.calculate_response(x=np.array([x]), channel_name=channel_name, get_sum=False)

        response_new = self.calculate_response(x=np.array([x + spend_delta]), channel_name=channel_name, get_sum=False)

        return float(response_new - response_current)

    def evaluate_total_response(
        self,
        x: np.ndarray,
        channel_names: List[str],
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """Evaluates total response across all channels."""
        print(f"\nTotal response evaluation:")
        print(f"Spend values: {x}")

        channel_responses = np.zeros(len(channel_names))
        gradients = np.zeros(len(channel_names))

        for i, channel in enumerate(channel_names):
            channel_responses[i] = self.calculate_response(x=np.array([x[i]]), channel_name=channel)
            # Calculate gradient analytically
            coef = self.coef_dict[channel]
            alpha = self.alphas_dict[f"{channel}_alphas"]
            gamma = self.gammas_dict[f"{channel}_gammas"]
            x_hist_carryover = np.mean(self.hist_carryover_dict[channel])
            x_adstocked = x[i] + x_hist_carryover

            hill_term = (gamma / x_adstocked) ** alpha
            gradients[i] = coef * alpha * hill_term / (x_adstocked * (1 + hill_term) ** 2)

        total_response = np.sum(channel_responses)

        print(f"Channel responses: {channel_responses}")
        print(f"Total response: {total_response}")
        print(f"Gradients: {gradients}")

        return total_response, channel_responses, gradients
