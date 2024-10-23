from typing import Dict, List
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
        self.hyperparameters = hyperparameters

    def _apply_geometric_adstock(self, values: pd.Series, theta: float) -> pd.Series:
        """
        Applies geometric adstock transformation.
        x[t] = x[t] + θ * x[t-1]
        """
        result = values.copy()
        for t in range(1, len(values)):
            result.iloc[t] += theta * result.iloc[t - 1]
        return result

    def _apply_weibull_adstock(self, values: pd.Series, shape: float, scale: float) -> pd.Series:
        """
        Applies Weibull adstock transformation.
        Uses shape and scale parameters to create decay curve.
        """
        max_lag = len(values)
        times = np.arange(max_lag)
        weights = (shape / scale) * (times / scale) ** (shape - 1) * np.exp(-((times / scale) ** shape))
        weights = weights / weights.sum()  # Normalize

        # Apply convolution
        return pd.Series(np.convolve(values, weights, mode="full")[: len(values)], index=values.index)

    def _apply_saturation(self, values: pd.Series, alpha: float, gamma: float) -> pd.Series:
        """
        Applies Hill function saturation transformation.
        S(x) = (x^alpha) / (x^alpha + gamma^alpha)
        """
        return (values**alpha) / (values**alpha + gamma**alpha)

    def apply_media_transforms(self, values: pd.Series, channel: str) -> pd.Series:
        """
        Applies adstock and saturation transformations to media values.

        Args:
            values: The media values to transform
            channel: The channel name to get correct hyperparameters

        Returns:
            pd.Series: Transformed values
        """
        try:
            channel_params = self.hyperparameters.get_hyperparameter(channel)

            if self.hyperparameters.adstock == AdstockType.GEOMETRIC:
                if not channel_params.thetas:
                    raise ValueError(f"Geometric adstock requires theta values for channel {channel}")
                theta = channel_params.thetas[0]  # Use first theta value
                transformed = self._apply_geometric_adstock(values, theta)
            else:
                if not (channel_params.shapes and channel_params.scales):
                    raise ValueError(f"Weibull adstock requires shape and scale values for channel {channel}")
                shape = channel_params.shapes[0]  # Use first shape value
                scale = channel_params.scales[0]  # Use first scale value
                transformed = self._apply_weibull_adstock(values, shape, scale)

            if not (channel_params.alphas and channel_params.gammas):
                raise ValueError(f"Saturation requires alpha and gamma values for channel {channel}")
            alpha = channel_params.alphas[0]  # Use first alpha value
            gamma = channel_params.gammas[0]  # Use first gamma value

            result = self._apply_saturation(transformed, alpha, gamma)

            # Ensure we return a Series with the same index
            if not isinstance(result, pd.Series):
                result = pd.Series(result, index=values.index)

            return result

        except Exception as e:
            raise ValueError(f"Error transforming values for channel {channel}: {str(e)}")

    def apply_carryover_effect(self, values: pd.Series) -> float:
        """
        Calculates total effect including carryover by summing
        all transformed values.

        Args:
            values: Series of transformed values

        Returns:
            float: Total effect value
        """
        if isinstance(values, pd.Series):
            return float(values.sum())
        return float(np.sum(values))
