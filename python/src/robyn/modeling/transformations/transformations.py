# pyre-strict

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from enum import Enum


class AdstockType(Enum):
    GEOMETRIC = "geometric"
    WEIBULL_CDF = "weibull_cdf"
    WEIBULL_PDF = "weibull_pdf"


class Transformations:
    @staticmethod
    def mic_men(x: np.ndarray, Vmax: float, Km: float, reverse: bool = False) -> np.ndarray:
        """
        Michaelis-Menten Transformation

        Args:
            x (np.ndarray): Input values
            Vmax (float): Maximum rate achieved by the system
            Km (float): Michaelis constant
            reverse (bool): If True, reverse the transformation

        Returns:
            np.ndarray: Transformed values
        """
        if not reverse:
            return Vmax * x / (Km + x)
        else:
            return x * Km / (Vmax - x)

    @staticmethod
    def adstock_geometric(x: np.ndarray, theta: float) -> Dict[str, Any]:
        """
        Geometric Adstocking

        Args:
            x (np.ndarray): Input values
            theta (float): Decay rate

        Returns:
            Dict[str, Any]: Dictionary containing transformed values and metadata
        """
        if len(x) > 1:
            x_decayed = np.zeros_like(x)
            x_decayed[0] = x[0]
            for xi in range(1, len(x_decayed)):
                x_decayed[xi] = x[xi] + theta * x_decayed[xi - 1]
            theta_vec_cum = np.array([theta**i for i in range(len(x))])
        else:
            x_decayed = x
            theta_vec_cum = np.array([theta])

        inflation_total = np.sum(x_decayed) / np.sum(x)
        return {"x": x, "x_decayed": x_decayed, "theta_vec_cum": theta_vec_cum, "inflation_total": inflation_total}

    @staticmethod
    def adstock_weibull(
        x: np.ndarray, shape: float, scale: float, windlen: int = None, type: str = "cdf"
    ) -> Dict[str, Any]:
        """
        Weibull Adstocking

        Args:
            x (np.ndarray): Input values
            shape (float): Shape parameter
            scale (float): Scale parameter
            windlen (int): Window length
            type (str): Type of Weibull function ("cdf" or "pdf")

        Returns:
            Dict[str, Any]: Dictionary containing transformed values and metadata
        """
        if windlen is None:
            windlen = len(x)

        if len(x) > 1:
            x_bin = np.arange(1, windlen + 1)
            scale_trans = int(np.quantile(x_bin, scale))

            if shape == 0 or scale == 0:
                x_decayed = x
                theta_vec_cum = theta_vec = np.zeros(windlen)
                x_imme = x
            else:
                if type.lower() == "cdf":
                    theta_vec = np.concatenate(([1], 1 - Transformations._weibull_cdf(x_bin[:-1], shape, scale_trans)))
                    theta_vec_cum = np.cumprod(theta_vec)
                elif type.lower() == "pdf":
                    theta_vec_cum = Transformations._normalize(Transformations._weibull_pdf(x_bin, shape, scale_trans))

                x_decayed = np.zeros(windlen)
                x_imme = np.zeros(len(x))
                for i, x_val in enumerate(x):
                    x_vec = np.concatenate((np.zeros(i), np.repeat(x_val, windlen - i)))
                    theta_vec_cum_lag = np.concatenate((np.zeros(i), theta_vec_cum[: (windlen - i)]))
                    x_prod = x_vec * theta_vec_cum_lag
                    x_decayed += x_prod
                    x_imme[i] = x_prod[i]
                x_decayed = x_decayed[: len(x)]
        else:
            x_decayed = x_imme = x
            theta_vec_cum = np.array([1])

        inflation_total = np.sum(x_decayed) / np.sum(x)
        return {
            "x": x,
            "x_decayed": x_decayed,
            "theta_vec_cum": theta_vec_cum,
            "inflation_total": inflation_total,
            "x_imme": x_imme,
        }

    @staticmethod
    def transform_adstock(
        x: np.ndarray,
        adstock: AdstockType,
        theta: Optional[float] = None,
        shape: Optional[float] = None,
        scale: Optional[float] = None,
        windlen: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Transform using specified adstock method

        Args:
            x (np.ndarray): Input values
            adstock (AdstockType): Type of adstock transformation
            theta (float): Theta parameter for geometric adstock
            shape (float): Shape parameter for Weibull adstock
            scale (float): Scale parameter for Weibull adstock
            windlen (int): Window length for Weibull adstock

        Returns:
            Dict[str, Any]: Dictionary containing transformed values and metadata
        """
        if windlen is None:
            windlen = len(x)

        if adstock == AdstockType.GEOMETRIC:
            return Transformations.adstock_geometric(x, theta)
        elif adstock == AdstockType.WEIBULL_CDF:
            return Transformations.adstock_weibull(x, shape, scale, windlen, "cdf")
        elif adstock == AdstockType.WEIBULL_PDF:
            return Transformations.adstock_weibull(x, shape, scale, windlen, "pdf")

    @staticmethod
    def saturation_hill(
        x: np.ndarray, alpha: float, gamma: float, x_marginal: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Hill Saturation Transformation

        Args:
            x (np.ndarray): Input values
            alpha (float): Alpha parameter
            gamma (float): Gamma parameter
            x_marginal (np.ndarray): Marginal input values

        Returns:
            np.ndarray: Transformed values
        """
        inflexion = np.dot(np.array([1 - gamma, gamma]), [np.min(x), np.max(x)])
        if x_marginal is None:
            return x**alpha / (x**alpha + inflexion**alpha)
        else:
            return x_marginal**alpha / (x_marginal**alpha + inflexion**alpha)

    @staticmethod
    def _weibull_cdf(x: np.ndarray, shape: float, scale: float) -> np.ndarray:
        return 1 - np.exp(-((x / scale) ** shape))

    @staticmethod
    def _weibull_pdf(x: np.ndarray, shape: float, scale: float) -> np.ndarray:
        return (shape / scale) * (x / scale) ** (shape - 1) * np.exp(-((x / scale) ** shape))

    @staticmethod
    def _normalize(x: np.ndarray) -> np.ndarray:
        if np.max(x) - np.min(x) == 0:
            return np.concatenate(([1], np.zeros(len(x) - 1)))
        else:
            return (x - np.min(x)) / (np.max(x) - np.min(x))

    @classmethod
    def run_transformations(cls, input_collect: Dict[str, Any], hyperparameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run all transformations on input data

        Args:
            input_collect (Dict[str, Any]): Input data and parameters
            hyperparameters (Dict[str, Any]): Hyperparameters for transformations

        Returns:
            Dict[str, Any]: Dictionary containing transformed data
        """
        # Implementation here
        pass
