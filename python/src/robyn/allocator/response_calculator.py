from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


class ResponseCurveCalculator:
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

        Implementation follows fx_objective from allocator.R
        """
        # Adstock scales
        x_adstocked = spend + np.mean(x_hist_carryover)

        # Hill transformation
        if get_sum:
            x_out = coef * np.sum((1 + inflexion**alpha / x_adstocked**alpha) ** -1)
        else:
            x_out = coef * ((1 + inflexion**alpha / x_adstocked**alpha) ** -1)

        return x_out

    def calculate_gradient(
        self, spend: float, coef: float, alpha: float, inflexion: float, x_hist_carryover: float = 0
    ) -> float:
        """
        Calculate gradient for optimization.

        Implementation follows fx_gradient from allocator.R
        """
        x_adstocked = spend + np.mean(x_hist_carryover)
        x_out = -coef * np.sum(
            (alpha * (inflexion**alpha) * (x_adstocked ** (alpha - 1))) / (x_adstocked**alpha + inflexion**alpha) ** 2
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
