from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from robyn.data.entities.mmmdata import MMMData
from robyn.data.entities.hyperparameters import Hyperparameters
from robyn.modeling.entities.pareto_result import ParetoResult


logger = logging.getLogger(__name__)


@dataclass
class HillParameters:
    """Parameters for hill transformation and response calculation."""

    alphas: np.ndarray
    gammas: np.ndarray
    coefs: np.ndarray
    carryover: np.ndarray


def get_hill_params(
    mmm_data: MMMData,
    hyperparameters: Hyperparameters,
    dt_hyppar: pd.DataFrame,
    dt_coef: pd.DataFrame,
    media_sorted: np.ndarray,
    select_model: str,
    media_vec_collect: pd.DataFrame = None,
) -> HillParameters:
    """Get Hill transformation parameters with proper coefficient mapping."""
    logger.debug("\nExtracting Hill parameters...")

    dt_coef_filtered = dt_coef[dt_coef["solID"] == select_model].set_index("rn")

    coefs = []
    alphas = []
    gammas = []
    carryover = []

    for channel in media_sorted:
        logger.debug(f"\nProcessing channel: {channel}")

        # Get alpha parameter
        alpha_col = f"{channel}_alphas"
        logger.debug(f"alpha_col: {alpha_col}")
        logger.debug("dt_hyppar: ", dt_hyppar)
        logger.debug("dt_hyppar alpha_col: ", dt_hyppar[alpha_col])
        alpha = dt_hyppar[alpha_col].iloc[0]
        alphas.append(alpha)
        logger.debug(f"Alpha: {alpha}")

        # Get gamma parameter
        gamma_col = f"{channel}_gammas"
        gamma = dt_hyppar[gamma_col].iloc[0]
        gammas.append(gamma)
        logger.debug(f"Gamma: {gamma}")

        # Get coefficient
        try:
            coef = float(dt_coef_filtered.loc[channel, "coef"])
        except KeyError:
            logger.debug(f"Warning: No coefficient found for {channel}, using 0.0")
            coef = 0.0
        coefs.append(coef)
        logger.debug(f"Coefficient: {coef}")

        # Calculate carryover to match R implementation
        if hyperparameters.adstock == "geometric":
            channel_params = hyperparameters.hyperparameters[channel]
            # Use mean instead of sum/2
            theta = np.mean(channel_params.thetas)
            carryover_val = geometric_adstock(theta)
        else:
            shape = np.mean(hyperparameters.hyperparameters[channel].shapes)
            scale = np.mean(hyperparameters.hyperparameters[channel].scales)
            carryover_val = weibull_adstock(shape, scale)

        carryover.append(carryover_val)
        logger.debug(f"Carryover: {carryover_val}")

        logger.debug(f"Final parameter set for {channel}:")
        logger.debug(f"  Alpha: {alpha}")
        logger.debug(f"  Gamma: {gamma}")
        logger.debug(f"  Coefficient: {coef}")
        logger.debug(f"  Carryover: {carryover_val}")

    return HillParameters(
        alphas=np.array(alphas),
        gammas=np.array(gammas),
        coefs=np.array(coefs),
        carryover=np.array(carryover),
    )


def calculate_carryover(
    mmm_data: MMMData,
    hyperparameters: Hyperparameters,
    media_vec_collect: pd.DataFrame,
    media_sorted: np.ndarray,
    select_model: str,
) -> np.ndarray:
    """
    Calculate carryover effects for each channel using hyperparameters directly.
    """
    carryover = np.zeros(len(media_sorted))

    logger.debug(f"\nCalculating carryover effects for channels: {media_sorted}")
    logger.debug(f"Adstock type: {hyperparameters.adstock}")

    # logger.debug available channels in hyperparameters
    logger.debug("\nAvailable channels in hyperparameters:")
    for channel, params in hyperparameters.hyperparameters.items():
        logger.debug(f"  {channel}:")
        logger.debug(f"    thetas: {params.thetas}")
        logger.debug(f"    alphas: {params.alphas}")
        logger.debug(f"    gammas: {params.gammas}")

    for i, channel in enumerate(media_sorted):
        logger.debug(f"\nProcessing channel: {channel}")
        try:
            if hyperparameters.adstock == "geometric":
                # Get channel parameters from hyperparameters dictionary
                channel_params = hyperparameters.hyperparameters[channel]
                if channel_params.thetas:
                    theta = sum(channel_params.thetas) / 2  # Use mean of min and max
                    logger.debug(
                        f"Found theta range: {channel_params.thetas}, using mean value: {theta}"
                    )
                    carryover[i] = geometric_adstock(theta)
                else:
                    logger.debug(f"No theta values found for {channel}")
                    logger.debug("Using default carryover effect of 0.1")
                    carryover[i] = 0.1
            else:  # weibull
                channel_params = hyperparameters.hyperparameters[channel]
                if channel_params.shapes and channel_params.scales:
                    shape = sum(channel_params.shapes) / 2
                    scale = sum(channel_params.scales) / 2
                    logger.debug(
                        f"Shape range: {channel_params.shapes}, Scale range: {channel_params.scales}"
                    )
                    logger.debug(f"Using mean values - Shape: {shape}, Scale: {scale}")
                    carryover[i] = weibull_adstock(shape, scale)
                else:
                    logger.debug(f"Missing shape/scale values for {channel}")
                    logger.debug("Using default carryover effect of 0.1")
                    carryover[i] = 0.1

            logger.debug(f"Calculated carryover effect: {carryover[i]}")

        except KeyError as e:
            logger.debug(f"Error: Channel {channel} not found in hyperparameters")
            logger.debug("Using default carryover effect of 0.1")
            carryover[i] = 0.1
        except Exception as e:
            logger.debug(f"Error processing channel {channel}: {str(e)}")
            logger.debug("Using default carryover effect of 0.1")
            carryover[i] = 0.1

    logger.debug("\nFinal carryover effects:")
    for ch, effect in zip(media_sorted, carryover):
        logger.debug(f"{ch}: {effect}")

    return carryover


def geometric_adstock(theta: float) -> float:
    if theta >= 1 or theta <= 0:
        return 0
    return theta / (1 - theta)


def weibull_adstock(shape: float, scale: float) -> float:
    """
    Calculate Weibull adstock effect with input validation.

    Args:
        shape: Shape parameter (must be positive)
        scale: Scale parameter (must be positive)

    Returns:
        Adstock effect value
    """
    if shape <= 0 or scale <= 0:
        return 0
    return scale * np.exp(-(1 / shape))


def calculate_gradient(
    spend: float, alpha: float, gamma: float, coef: float, carryover: float
) -> float:
    """
    Calculate gradient of response function for a given spend value.

    Args:
        spend: Spend value
        alpha: Hill function alpha parameter
        gamma: Hill function gamma parameter
        coef: Channel coefficient
        carryover: Carryover effect

    Returns:
        Gradient value
    """
    x_adstocked = spend + carryover
    numerator = alpha * (gamma**alpha) * (x_adstocked ** (alpha - 1))
    denominator = (x_adstocked**alpha + gamma**alpha) ** 2
    return -coef * numerator / denominator


def check_allocator_constraints(
    channel_constr_low: np.ndarray, channel_constr_up: np.ndarray
) -> None:
    """
    Validate allocator constraints.

    Args:
        channel_constr_low: Lower bounds for channel constraints
        channel_constr_up: Upper bounds for channel constraints

    Raises:
        ValueError: If constraints are invalid
    """
    if np.any(channel_constr_low < 0):
        raise ValueError("Lower bounds must be non-negative")

    if np.any(channel_constr_up <= 0):
        raise ValueError("Upper bounds must be positive")

    if len(channel_constr_low) > 1 and len(channel_constr_up) > 1:
        if len(channel_constr_low) != len(channel_constr_up):
            raise ValueError("Lower and upper bounds must have same length")

        if np.any(channel_constr_up < channel_constr_low):
            raise ValueError("Upper bounds must be greater than lower bounds")


def check_metric_dates(
    date_range: str, dates: pd.Series, interval: int, is_allocator: bool = True
) -> Dict[str, Union[List[datetime], int]]:
    """
    Check and process date ranges for metrics calculation.

    Args:
        date_range: Date range specification ('all', 'last', or date range)
        dates: Series of dates
        interval: Time interval in days
        is_allocator: Whether function is called from allocator

    Returns:
        Dictionary containing processed date range and interval information
    """
    dates = pd.to_datetime(dates)
    date_min = dates.min()
    date_max = dates.max()

    if date_range == "all":
        selected_dates = dates.tolist()
    elif date_range == "last":
        selected_dates = [date_max]
    elif isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        range_start = pd.to_datetime(date_range[0])
        range_end = pd.to_datetime(date_range[1])

        if range_start < date_min or range_end > date_max:
            raise ValueError("Date range must be within available data dates")

        selected_dates = dates[(dates >= range_start) & (dates <= range_end)].tolist()
    else:
        try:
            single_date = pd.to_datetime(date_range)
            if single_date < date_min or single_date > date_max:
                raise ValueError("Date must be within available data dates")
            selected_dates = [single_date]
        except:
            raise ValueError("Invalid date range format")

    return {"date_range_updated": selected_dates, "interval": interval}
