from typing import List, Dict, Union, Optional
import numpy as np
import pandas as pd

class AdstockTransformation:
    def __init__(self) -> None:
        pass

    def mic_men(self, x: float, Vmax: float, Km: float, reverse: bool = False) -> float:
        """
        Calculate the Michaelis-Menten transformation.

        Args:
            x: The input value.
            Vmax: The maximum rate of the transformation.
            Km: The Michaelis constant.
            reverse: Whether to perform the reverse transformation.

        Returns:
            The transformed value.
        """
        pass

    def adstock_geometric(self, x: List[float], theta: float) -> Dict[str, Union[List[float], float]]:
        """
        Adstock geometric function

        Args:
            x: The input values.
            theta: The decay factor.

        Returns:
            A dictionary containing the original input values, the decayed values,
            the cumulative decay factors, and the inflation total.
        """
        pass

    def adstock_weibull(self, x: List[float], shape: float, scale: float, windlen: Optional[int] = None, stype: str = "cdf") -> Dict[str, Union[List[float], float, np.ndarray]]:
        """
        Adstock Weibull function

        Args:
            x: The input time series data.
            shape: The shape parameter of the Weibull distribution.
            scale: The scale parameter of the Weibull distribution.
            windlen: The length of the adstock window.
            stype: The type of adstock transformation to perform.

        Returns:
            A dictionary containing the transformed data and related information.
        """
        pass

    def transform_adstock(self, x: List[float], adstock: Literal["geometric", "weibull_cdf", "weibull_pdf"], theta: Optional[float] = None, shape: Optional[float] = None, scale: Optional[float] = None, windlen: Optional[int] = None) -> Dict[str, Union[List[float], float, np.ndarray]]:
        """
        Transforms the input data using the adstock model.

        Args:
            x: The input data to be transformed.
            adstock: The type of adstock model to be applied.
            theta: The decay factor for the geometric adstock model.
            shape: The shape parameter for the Weibull adstock model.
            scale: The scale parameter for the Weibull adstock model.
            windlen: The length of the adstock window.

        Returns:
            The transformed data based on the adstock model.
        """
        pass

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """
        Normalizes the input data.

        Args:
            x: The input data to be normalized.

        Returns:
            The normalized data.
        """
        pass

    def saturation_hill(self, x: np.ndarray, alpha: float, gamma: float, x_marginal: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Implements the saturation hill function.

        Args:
            x: Input values.
            alpha: Exponent parameter.
            gamma: Weighting parameter.
            x_marginal: Optional marginal values.

        Returns:
            Output values computed using the saturation hill function.
        """
        pass
