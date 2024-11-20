from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from robyn.data.entities.mmmdata import MMMData
from robyn.data.entities.hyperparameters import Hyperparameters
from robyn.modeling.entities.pareto_result import ParetoResult


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
    media_vec_collect: pd.DataFrame,
) -> HillParameters:
    """
    Get Hill transformation parameters for each channel.

    Args:
        mmm_data: Marketing Mix Model data
        hyperparameters: Model hyperparameters
        dt_hyppar: Hyperparameters for selected model
        dt_coef: Coefficients for selected model
        media_sorted: Sorted media channel names
        select_model: Selected model ID
        media_vec_collect: Collected media vectors

    Returns:
        HillParameters object containing transformation parameters
    """
    # Extract hill parameters from hyperparameters
    alphas = []
    gammas = []
    for channel in media_sorted:
        alpha_col = f"{channel}_alphas"
        gamma_col = f"{channel}_gammas"

        alpha = dt_hyppar[alpha_col].iloc[0] if alpha_col in dt_hyppar else 1.0
        gamma = dt_hyppar[gamma_col].iloc[0] if gamma_col in dt_hyppar else 0.5

        alphas.append(alpha)
        gammas.append(gamma)

    # Get coefficients for each channel
    coefs = dt_coef[dt_coef["rn"].isin(media_sorted)]["coef"].values

    # Calculate carryover effects
    carryover = calculate_carryover(
        mmm_data, hyperparameters, media_vec_collect, media_sorted, select_model
    )

    return HillParameters(
        alphas=np.array(alphas),
        gammas=np.array(gammas),
        coefs=np.array(coefs),
        carryover=carryover,
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

    print(f"\nCalculating carryover effects for channels: {media_sorted}")
    print(f"Adstock type: {hyperparameters.adstock}")

    # Print available channels in hyperparameters
    print("\nAvailable channels in hyperparameters:")
    for channel, params in hyperparameters.hyperparameters.items():
        print(f"  {channel}:")
        print(f"    thetas: {params.thetas}")
        print(f"    alphas: {params.alphas}")
        print(f"    gammas: {params.gammas}")

    for i, channel in enumerate(media_sorted):
        print(f"\nProcessing channel: {channel}")
        try:
            if hyperparameters.adstock == "geometric":
                # Get channel parameters from hyperparameters dictionary
                channel_params = hyperparameters.hyperparameters[channel]
                if channel_params.thetas:
                    theta = sum(channel_params.thetas) / 2  # Use mean of min and max
                    print(
                        f"Found theta range: {channel_params.thetas}, using mean value: {theta}"
                    )
                    carryover[i] = geometric_adstock(theta)
                else:
                    print(f"No theta values found for {channel}")
                    print("Using default carryover effect of 0.1")
                    carryover[i] = 0.1
            else:  # weibull
                channel_params = hyperparameters.hyperparameters[channel]
                if channel_params.shapes and channel_params.scales:
                    shape = sum(channel_params.shapes) / 2
                    scale = sum(channel_params.scales) / 2
                    print(
                        f"Shape range: {channel_params.shapes}, Scale range: {channel_params.scales}"
                    )
                    print(f"Using mean values - Shape: {shape}, Scale: {scale}")
                    carryover[i] = weibull_adstock(shape, scale)
                else:
                    print(f"Missing shape/scale values for {channel}")
                    print("Using default carryover effect of 0.1")
                    carryover[i] = 0.1

            print(f"Calculated carryover effect: {carryover[i]}")

        except KeyError as e:
            print(f"Error: Channel {channel} not found in hyperparameters")
            print("Using default carryover effect of 0.1")
            carryover[i] = 0.1
        except Exception as e:
            print(f"Error processing channel {channel}: {str(e)}")
            print("Using default carryover effect of 0.1")
            carryover[i] = 0.1

    print("\nFinal carryover effects:")
    for ch, effect in zip(media_sorted, carryover):
        print(f"{ch}: {effect}")

    return carryover


def geometric_adstock(theta: float) -> float:
    """
    Calculate geometric adstock effect with improved calculation.

    Args:
        theta: Decay parameter (between 0 and 1)

    Returns:
        Adstock effect value
    """
    if theta >= 1:
        return 0
    if theta <= 0:
        return 0
    return theta / (1 - theta)


def calculate_response(
    spend: float, alpha: float, gamma: float, coef: float, carryover: float
) -> float:
    """
    Calculate response with debug output.
    """
    x_adstocked = spend + carryover
    response = coef * (1 / (1 + (gamma**alpha / x_adstocked**alpha)))
    print(f"\nResponse calculation debug:")
    print(f"  Input spend: {spend:.2f}")
    print(f"  Carryover effect: {carryover:.4f}")
    print(f"  Total adstocked value: {x_adstocked:.2f}")
    print(f"  Alpha: {alpha:.4f}")
    print(f"  Gamma: {gamma:.4f}")
    print(f"  Coefficient: {coef:.4f}")
    print(f"  Calculated response: {response:.4f}")
    print(f"  Response per spend unit: {response/spend if spend > 0 else 0:.4f}")
    return response


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
