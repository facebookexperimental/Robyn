from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


class ResponseCalculator:
    """Calculates response curves and related metrics."""

    def calculate_response(
        self,
        spend: float,
        coef: float,
        alpha: float,
        inflexion: float,
        x_hist_carryover: float = 0,
        get_sum: bool = True,
    ) -> float:
        """
        Calculate response for given spend level and parameters.

        Args:
            spend: Spend value
            coef: Response coefficient
            alpha: Alpha parameter for hill function
            inflexion: Inflexion point parameter
            x_hist_carryover: Historical carryover value
            get_sum: Whether to sum the response values

        Returns:
            float: Calculated response value
        """
        eps = 1e-10  # Small epsilon to prevent division by zero

        # Add epsilon to prevent zero division
        x_adstocked = spend + np.mean(x_hist_carryover) + eps

        # Hill transformation
        if get_sum:
            x_out = coef * np.sum((1 + (inflexion + eps) ** alpha / (x_adstocked) ** alpha) ** -1)
        else:
            x_out = coef * ((1 + (inflexion + eps) ** alpha / (x_adstocked) ** alpha) ** -1)

        return x_out

    def calculate_gradient(
        self, spend: float, coef: float, alpha: float, inflexion: float, x_hist_carryover: float = 0
    ) -> float:
        """Calculate gradient for optimization."""
        eps = 1e-10
        x_adstocked = spend + np.mean(x_hist_carryover) + eps

        x_out = -coef * np.sum(
            (alpha * ((inflexion + eps) ** alpha) * (x_adstocked ** (alpha - 1)))
            / ((x_adstocked**alpha + (inflexion + eps) ** alpha) ** 2)
        )
        return x_out

    def get_response_curve(
        self, spend_range: np.ndarray, coef: float, alpha: float, inflexion: float, x_hist_carryover: float = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate response curve over a range of spend values."""
        responses = np.array(
            [
                self.calculate_response(spend, coef, alpha, inflexion, x_hist_carryover, get_sum=False)
                for spend in spend_range
            ]
        )

        return spend_range, responses
