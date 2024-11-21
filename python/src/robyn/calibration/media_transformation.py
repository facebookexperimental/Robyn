# pyre-strict

import logging

import pandas as pd
import numpy as np
from robyn.data.entities.hyperparameters import Hyperparameters
from robyn.data.entities.enums import AdstockType


class MediaTransformation:
    """
    Handles media transformations including adstock and saturation effects.
    """

    def __init__(self, hyperparameters: Hyperparameters):
        """Initialize with model hyperparameters."""
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing MediaTransformation with hyperparameters")
        self.logger.debug("Hyperparameters configuration: %s", hyperparameters)
        self.hyperparameters = hyperparameters

    def _apply_geometric_adstock(self, values: pd.Series, theta: float) -> pd.Series:
        """
        Applies geometric adstock transformation.
        x[t] = x[t] + θ * x[t-1]
        """
        self.logger.debug(
            "Starting geometric adstock transformation with theta=%f", theta
        )
        self.logger.debug("Input series shape: %s", values.shape)

        result = values.copy()
        for t in range(1, len(values)):
            result.iloc[t] += theta * result.iloc[t - 1]

        self.logger.debug("Completed geometric adstock transformation")
        return result

    def apply_weibull_adstock(
        self, values: pd.Series, shape: float, scale: float
    ) -> pd.Series:
        """
        Applies Weibull adstock transformation.
        Uses shape and scale parameters to create decay curve.
        """
        self.logger.debug(
            "Starting Weibull adstock transformation with shape=%f, scale=%f",
            shape,
            scale,
        )
        self.logger.debug("Input series shape: %s", values.shape)

        max_lag = len(values)
        times = np.arange(max_lag)
        weights = (
            (shape / scale)
            * (times / scale) ** (shape - 1)
            * np.exp(-((times / scale) ** shape))
        )
        weights = weights / weights.sum()  # Normalize

        self.logger.debug("Calculated Weibull weights, proceeding with convolution")
        result = pd.Series(
            np.convolve(values, weights, mode="full")[: len(values)], index=values.index
        )

        self.logger.debug("Completed Weibull adstock transformation")
        return result

    def apply_saturation(
        self, values: pd.Series, alpha: float, gamma: float
    ) -> pd.Series:
        """
        Applies Hill function saturation transformation.
        S(x) = (x^alpha) / (x^alpha + gamma^alpha)
        """
        self.logger.debug(
            "Starting saturation transformation with alpha=%f, gamma=%f", alpha, gamma
        )
        self.logger.debug("Input series shape: %s", values.shape)

        result = (values**alpha) / (values**alpha + gamma**alpha)

        self.logger.debug("Completed saturation transformation")
        return result

    def apply_media_transforms(self, values: pd.Series, channel: str) -> pd.Series:
        """
        Applies adstock and saturation transformations to media values.

        Args:
            values: The media values to transform
            channel: The channel name to get correct hyperparameters

        Returns:
            pd.Series: Transformed values
        """
        self.logger.info("Starting media transformation for channel: %s", channel)
        self.logger.debug("Input values shape: %s", values.shape)

        try:
            channel_params = self.hyperparameters.get_hyperparameter(channel)
            self.logger.debug(
                "Retrieved hyperparameters for channel %s: %s", channel, channel_params
            )

            if self.hyperparameters.adstock == AdstockType.GEOMETRIC:
                if not channel_params.thetas:
                    error_msg = (
                        f"Geometric adstock requires theta values for channel {channel}"
                    )
                    self.logger.error(error_msg)
                    raise ValueError(error_msg)

                theta = channel_params.thetas[0]  # Use first theta value
                self.logger.info("Applying geometric adstock transformation")
                transformed = self._apply_geometric_adstock(values, theta)
            else:
                if not (channel_params.shapes and channel_params.scales):
                    error_msg = f"Weibull adstock requires shape and scale values for channel {channel}"
                    self.logger.error(error_msg)
                    raise ValueError(error_msg)

                shape = channel_params.shapes[0]  # Use first shape value
                scale = channel_params.scales[0]  # Use first scale value
                self.logger.info("Applying Weibull adstock transformation")
                transformed = self._apply_weibull_adstock(values, shape, scale)

            if not (channel_params.alphas and channel_params.gammas):
                error_msg = (
                    f"Saturation requires alpha and gamma values for channel {channel}"
                )
                self.logger.error(error_msg)
                raise ValueError(error_msg)

            alpha = channel_params.alphas[0]  # Use first alpha value
            gamma = channel_params.gammas[0]  # Use first gamma value

            self.logger.info("Applying saturation transformation")
            result = self._apply_saturation(transformed, alpha, gamma)

            # Ensure we return a Series with the same index
            if not isinstance(result, pd.Series):
                self.logger.debug("Converting result to pandas Series")
                result = pd.Series(result, index=values.index)

            self.logger.info("Completed media transformation for channel: %s", channel)
            self.logger.debug("Output shape: %s", result.shape)
            return result

        except Exception as e:
            error_msg = f"Error transforming values for channel {channel}: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            raise ValueError(error_msg)

    def apply_carryover_effect(self, values: pd.Series) -> float:
        """
        Calculates total effect including carryover by summing
        all transformed values.

        Args:
            values: Series of transformed values

        Returns:
            float: Total effect value
        """
        self.logger.debug("Calculating carryover effect")
        self.logger.debug(
            "Input values shape: %s",
            values.shape if isinstance(values, pd.Series) else np.array(values).shape,
        )

        result = float(
            values.sum() if isinstance(values, pd.Series) else np.sum(values)
        )

        self.logger.debug("Calculated carryover effect: %f", result)
        return result
