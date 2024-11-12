#pyre-strict

import logging
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)

class ResponseCalculator:
    """Calculates response curves and related metrics."""

    def __init__(self):
        """Initialize ResponseCalculator with logging setup."""
        logger.debug("Initializing ResponseCalculator instance")

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
        logger.debug(
            "Starting response calculation with parameters: spend=%.2f, coef=%.2f, alpha=%.2f, "
            "inflexion=%.2f, x_hist_carryover=%.2f, get_sum=%s",
            spend, coef, alpha, inflexion, x_hist_carryover, get_sum
        )

        eps = 1e-10  # Small epsilon to prevent division by zero

        # Add epsilon to prevent zero division
        x_adstocked = spend + np.mean(x_hist_carryover) + eps
        logger.debug("Calculated x_adstocked value: %.4f", x_adstocked)

        try:
            # Hill transformation
            if get_sum:
                x_out = coef * np.sum((1 + (inflexion + eps) ** alpha / (x_adstocked) ** alpha) ** -1)
                logger.debug("Calculated summed response: %.4f", x_out)
            else:
                x_out = coef * ((1 + (inflexion + eps) ** alpha / (x_adstocked) ** alpha) ** -1)
                logger.debug("Calculated individual response: %.4f", x_out)

        except Exception as e:
            logger.error("Error during response calculation: %s", str(e))
            raise

        if x_out < 0:
            logger.warning("Negative response value calculated: %.4f", x_out)

        logger.info("Successfully calculated response value: %.4f", x_out)
        return x_out

    def calculate_gradient(
        self, spend: float, coef: float, alpha: float, inflexion: float, x_hist_carryover: float = 0
    ) -> float:
        """Calculate gradient for optimization."""
        logger.debug(
            "Starting gradient calculation with parameters: spend=%.2f, coef=%.2f, alpha=%.2f, "
            "inflexion=%.2f, x_hist_carryover=%.2f",
            spend, coef, alpha, inflexion, x_hist_carryover
        )

        eps = 1e-10
        try:
            x_adstocked = spend + np.mean(x_hist_carryover) + eps
            logger.debug("Calculated x_adstocked value: %.4f", x_adstocked)

            x_out = -coef * np.sum(
                (alpha * ((inflexion + eps) ** alpha) * (x_adstocked ** (alpha - 1)))
                / ((x_adstocked**alpha + (inflexion + eps) ** alpha) ** 2)
            )

            if abs(x_out) < eps:
                logger.warning("Gradient value near zero: %.4e", x_out)

            logger.debug("Calculated gradient value: %.4f", x_out)

        except Exception as e:
            logger.error("Error during gradient calculation: %s", str(e))
            raise

        logger.info("Successfully calculated gradient value: %.4f", x_out)
        return x_out

    def get_response_curve(
        self, spend_range: np.ndarray, coef: float, alpha: float, inflexion: float, x_hist_carryover: float = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate response curve over a range of spend values."""
        logger.info(
            "Generating response curve with parameters: coef=%.2f, alpha=%.2f, inflexion=%.2f, x_hist_carryover=%.2f",
            coef, alpha, inflexion, x_hist_carryover
        )
        logger.debug("Spend range: min=%.2f, max=%.2f, size=%d", 
                    np.min(spend_range), np.max(spend_range), len(spend_range))

        try:
            responses = np.array(
                [
                    self.calculate_response(spend, coef, alpha, inflexion, x_hist_carryover, get_sum=False)
                    for spend in spend_range
                ]
            )
            
            logger.debug("Response range: min=%.2f, max=%.2f", np.min(responses), np.max(responses))

            if np.any(np.isnan(responses)):
                logger.warning("NaN values detected in response curve calculations")

            if np.any(responses < 0):
                logger.warning("Negative values detected in response curve calculations")

        except Exception as e:
            logger.error("Error generating response curve: %s", str(e))
            raise

        logger.info("Successfully generated response curve with %d points", len(responses))
        return spend_range, responses