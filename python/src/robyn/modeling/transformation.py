import numpy as np
import pandas as pd
from typing import Dict, List, Union

def mic_men(x: Union[float, np.ndarray], Vmax: float, Km: float, reverse: bool = False) -> Union[float, np.ndarray]:
    """Michaelis-Menten Transformation"""
    if not reverse:
        return Vmax * x / (Km + x)
    else:
        return x * Km / (Vmax - x)

def adstock_geometric(x: np.ndarray, theta: float) -> Dict[str, Union[np.ndarray, float]]:
    """Geometric Adstock Transformation"""
    x_decayed = np.zeros_like(x)
    x_decayed[0] = x[0]
    for xi in range(1, len(x)):
        x_decayed[xi] = x[xi] + theta * x_decayed[xi - 1]
    
    theta_vec_cum = np.array([theta**i for i in range(len(x))])
    inflation_total = np.sum(x_decayed) / np.sum(x)
    
    return {
        "x": x,
        "x_decayed": x_decayed,
        "thetaVecCum": theta_vec_cum,
        "inflation_total": inflation_total
    }

def adstock_weibull(x: np.ndarray, shape: float, scale: float, windlen: int = None, type: str = "cdf") -> Dict[str, Union[np.ndarray, float]]:
    """Weibull Adstock Transformation"""
    if windlen is None:
        windlen = len(x)
    
    if shape == 0 or scale == 0:
        return {
            "x": x,
            "x_decayed": x,
            "thetaVecCum": np.zeros(windlen),
            "inflation_total": 1,
            "x_imme": x
        }
    
    x_bin = np.arange(1, windlen + 1)
    scale_trans = int(np.quantile(x_bin, scale))
    
    if type.lower() == "cdf":
        theta_vec = np.concatenate(([1], 1 - weibull_cdf(x_bin[:-1], shape, scale_trans)))
        theta_vec_cum = np.cumprod(theta_vec)
    elif type.lower() == "pdf":
        theta_vec_cum = normalize(weibull_pdf(x_bin, shape, scale_trans))
    else:
        raise ValueError("Type must be 'cdf' or 'pdf'")
    
    x_decayed = np.zeros(windlen)
    x_imme = np.zeros(windlen)
    
    for i in range(len(x)):
        x_vec = np.concatenate((np.zeros(i), np.repeat(x[i], windlen - i)))
        theta_vec_cum_lag = np.concatenate((np.zeros(i), theta_vec_cum[:windlen-i]))
        x_prod = x_vec * theta_vec_cum_lag
        x_decayed += x_prod
        x_imme[i] = x_prod[i]
    
    inflation_total = np.sum(x_decayed) / np.sum(x)
    
    return {
        "x": x,
        "x_decayed": x_decayed,
        "thetaVecCum": theta_vec_cum,
        "inflation_total": inflation_total,
        "x_imme": x_imme
    }

def weibull_cdf(x: np.ndarray, shape: float, scale: float) -> np.ndarray:
    return 1 - np.exp(-(x / scale) ** shape)

def weibull_pdf(x: np.ndarray, shape: float, scale: float) -> np.ndarray:
    return (shape / scale) * (x / scale)**(shape - 1) * np.exp(-(x / scale)**shape)

def normalize(x: np.ndarray) -> np.ndarray:
    return (x - np.min(x)) / (np.max(x) - np.min(x))

def transform_adstock(x: np.ndarray, adstock: str, theta: float = None, shape: float = None, scale: float = None, windlen: int = None) -> Dict[str, Union[np.ndarray, float]]:
    """General Adstock Transformation function"""
    if adstock == "geometric":
        return adstock_geometric(x, theta)
    elif adstock == "weibull_cdf":
        return adstock_weibull(x, shape, scale, windlen, type="cdf")
    elif adstock == "weibull_pdf":
        return adstock_weibull(x, shape, scale, windlen, type="pdf")
    else:
        raise ValueError("Invalid adstock type")

def saturation_hill(x: np.ndarray, alpha: float, gamma: float, x_marginal: float = None) -> np.ndarray:
    """Hill Saturation Transformation"""
    inflection = np.dot([1 - gamma, gamma], [np.min(x), np.max(x)])
    if x_marginal is None:
        return x**alpha / (x**alpha + inflection**alpha)
    else:
        return x_marginal**alpha / (x_marginal**alpha + inflection**alpha)

# Add plotting functions (plot_adstock, plot_saturation) here if needed
