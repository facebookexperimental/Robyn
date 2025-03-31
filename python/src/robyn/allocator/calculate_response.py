from typing import Optional, List, Dict, Any, Union
import pandas as pd
import numpy as np
from .checks import (
    check_metric_dates,
    check_metric_type,
    check_metric_value,
    check_adstock,
)
from scipy.stats import weibull_min
from scipy.stats import norm
from robyn.data.entities.mmmdata import MMMData
from robyn.modeling.entities.modeloutputs import ModelOutputs, Trial
from robyn.modeling.entities.modelrun_trials_config import TrialsConfig
from robyn.modeling.entities.model_refit_output import ModelRefitOutput
from robyn.modeling.feature_engineering import FeaturizedMMMData
from robyn.data.entities.hyperparameters import Hyperparameters
from robyn.modeling.entities.pareto_result import ParetoResult


def which_usecase(
    metric_value: Optional[Union[float, List[float]]],
    date_range: Optional[Union[str, List[str]]],
) -> str:
    """Determine the use case based on metric value and date range.

    Args:
        metric_value: Single value or list of values for the metric
        date_range: Date range specification

    Returns:
        String indicating the use case
    """
    if metric_value is None and date_range is None:
        return "all_historical_vec"
    elif metric_value is None and date_range is not None:
        return "selected_historical_vec"
    elif isinstance(metric_value, (int, float)) and date_range is None:
        return "total_metric_default_range"
    elif isinstance(metric_value, (int, float)) and date_range is not None:
        return "total_metric_selected_range"
    elif isinstance(metric_value, (list, np.ndarray, pd.Series)) and date_range is None:
        return "unit_metric_default_last_n"

    # Check for "all" date range
    if isinstance(date_range, str) and date_range == "all":
        return "all_historical_vec"

    return "unit_metric_selected_dates"


def mic_men(x: float, Vmax: float, Km: float, reverse: bool = False) -> float:
    """Michaelis-Menten transformation.

    Args:
        x: Input value
        Vmax: Maximum velocity
        Km: Michaelis constant
        reverse: Whether to reverse the transformation

    Returns:
        Transformed value
    """
    if not reverse:
        mm_out = (Vmax * x) / (Km + x)
    else:
        mm_out = spend = (x * Km) / (Vmax - x)

    return mm_out


def robyn_response(
    mmm_data: MMMData,
    pareto_result: ParetoResult,
    hyperparameters: Hyperparameters,
    featurized_mmm_data: FeaturizedMMMData,
    select_model: str,
    metric_name: str,
    metric_value: Optional[Union[float, List[float]]] = None,
    date_range: Optional[Union[str, List[str]]] = None,
    dt_hyppar: Optional[pd.DataFrame] = None,
    dt_coef: Optional[pd.DataFrame] = None,
    quiet: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Calculate response metrics for media channels."""
    # Get hyperparameters and coefficients from pareto_result if not provided
    if dt_hyppar is None:
        dt_hyppar = pareto_result.result_hyp_param
    if dt_coef is None:
        dt_coef = pareto_result.x_decomp_agg

    # Validate inputs
    if any(x is None for x in [dt_hyppar, dt_coef, mmm_data, pareto_result]):
        raise ValueError("InputCollect & OutputCollect must be provided")

    # Check if select_model should be updated from pareto_result
    if hasattr(pareto_result, "select_id"):
        select_model = pareto_result.select_id

    # Prep environment
    dt_input = mmm_data.data
    start_rw = mmm_data.mmmdata_spec.rolling_window_start_which
    end_rw = mmm_data.mmmdata_spec.rolling_window_end_which
    adstock = hyperparameters.adstock
    spend_expo_mod = featurized_mmm_data.modNLS["results"]
    paid_media_vars = mmm_data.mmmdata_spec.paid_media_vars
    paid_media_spends = mmm_data.mmmdata_spec.paid_media_spends
    exposure_vars = mmm_data.mmmdata_spec.exposure_vars
    organic_vars = mmm_data.mmmdata_spec.organic_vars
    all_solutions = dt_hyppar["solID"].unique()
    day_interval = mmm_data.mmmdata_spec.day_interval

    # Validate select_model
    if select_model not in all_solutions or select_model is None:
        raise ValueError(
            f"Input 'select_model' must be one of these values: {', '.join(map(str, all_solutions))}"
        )

    # Get use case based on inputs
    usecase = which_usecase(metric_value, date_range)

    # Check inputs with usecases
    metric_type = check_metric_type(
        metric_name, paid_media_spends, paid_media_vars, exposure_vars, organic_vars
    )

    # Get dates from DataFrame directly as a Series
    all_dates = dt_input[mmm_data.mmmdata_spec.date_var]  # This will be a pandas Series
    all_values = pd.Series(dt_input[metric_name].values)  # Keep this as numpy array

    print("date_var", mmm_data.mmmdata_spec.date_var)
    print("data.columns", mmm_data.data.columns)
    print("all_dates", all_dates)
    print("all_values", all_values)
    # Handle different use cases
    if usecase == "all_historical_vec":
        ds_list = check_metric_dates(
            date_range="all",
            all_dates=all_dates[:end_rw],  # Now passing a pandas Series
            day_interval=day_interval,
            quiet=quiet,
            **kwargs,
        )
        metric_value = None
    elif usecase == "unit_metric_default_last_n":
        ds_list = check_metric_dates(
            date_range=f"last_{len(metric_value)}",
            all_dates=all_dates[:end_rw],
            day_interval=day_interval,
            quiet=quiet,
            **kwargs,
        )
    else:
        ds_list = check_metric_dates(
            date_range=date_range,
            all_dates=all_dates[:end_rw],
            day_interval=day_interval,
            quiet=quiet,
            **kwargs,
        )

    # Before check_metric_value call
    print("\nDebug info before check_metric_value:")
    print(f"metric_value type: {type(metric_value)}")
    print(f"metric_name: {metric_name}")
    print(f"all_values type: {type(all_values)}")
    print(
        f"all_values shape: {all_values.shape if hasattr(all_values, 'shape') else len(all_values)}"
    )
    print(f"metric_loc type: {type(ds_list['metric_loc'])}")
    print(f"metric_loc: {ds_list['metric_loc'][:5]}...")  # Show first 5 elements

    # Check metric values
    val_list = check_metric_value(
        metric_value, metric_name, all_values, ds_list["metric_loc"]
    )
    date_range_updated = ds_list["date_range_updated"]
    metric_value_updated = val_list["metric_value_updated"]
    all_values_updated = val_list["all_values_updated"]

    # Transform exposure to spend when necessary
    if metric_type == "exposure":
        # Get corresponding spend name
        get_spend_name = paid_media_spends[paid_media_vars.index(metric_name)]

        # Check if spend exposure model exists
        if spend_expo_mod is None:
            raise ValueError(
                "Can't calculate exposure to spend response. Please, recreate your InputCollect object"
            )

        # Filter for current channel
        temp = spend_expo_mod[spend_expo_mod["channel"] == metric_name]
        nls_select = temp["rsq_nls"].iloc[0] > temp["rsq_lm"].iloc[0]

        if nls_select:
            # Use Michaelis-Menten transformation
            Vmax = temp.loc[temp["channel"] == metric_name, "Vmax"].iloc[0]
            Km = temp.loc[temp["channel"] == metric_name, "Km"].iloc[0]
            input_immediate = mic_men(
                x=metric_value_updated, Vmax=Vmax, Km=Km, reverse=True
            )
        else:
            # Use linear transformation
            coef_lm = temp.loc[temp["channel"] == metric_name, "coef_lm"].iloc[0]
            input_immediate = metric_value_updated / coef_lm

        # Update values
        all_values_updated.iloc[ds_list["metric_loc"]] = input_immediate
        hpm_name = get_spend_name
    else:
        input_immediate = metric_value_updated
        hpm_name = metric_name

    # Adstocking original
    media_vec_origin = dt_input[metric_name].values
    theta = scale = shape = None

    if adstock == "geometric":
        theta = dt_hyppar.loc[
            dt_hyppar["solID"] == select_model, f"{hpm_name}_thetas"
        ].iloc[0]

    if "weibull" in adstock:
        shape = dt_hyppar.loc[
            dt_hyppar["solID"] == select_model, f"{hpm_name}_shapes"
        ].iloc[0]
        scale = dt_hyppar.loc[
            dt_hyppar["solID"] == select_model, f"{hpm_name}_scales"
        ].iloc[0]

    x_list = transform_adstock(
        media_vec_origin, adstock, theta=theta, shape=shape, scale=scale
    )
    m_adstocked = x_list["x_decayed"]
    # net_carryover_ref = m_adstocked - media_vec_origin

    # Adstocking simulation
    x_list_sim = transform_adstock(
        all_values_updated, adstock, theta=theta, shape=shape, scale=scale
    )
    media_vec_sim = x_list_sim["x_decayed"]
    media_vec_sim_imme = (
        x_list_sim["x_imme"] if adstock == "weibull_pdf" else x_list_sim["x"]
    )
    input_total = media_vec_sim[ds_list["metric_loc"]]
    input_immediate = media_vec_sim_imme[ds_list["metric_loc"]]
    input_carryover = input_total - input_immediate

    # Saturation
    m_adstocked_rw = m_adstocked[start_rw:end_rw]
    alpha = dt_hyppar.loc[
        dt_hyppar["solID"] == select_model, f"{hpm_name}_alphas"
    ].iloc[0]
    gamma = dt_hyppar.loc[
        dt_hyppar["solID"] == select_model, f"{hpm_name}_gammas"
    ].iloc[0]

    if usecase == "all_historical_vec":
        metric_saturated_total = saturation_hill(
            x=m_adstocked_rw, alpha=alpha, gamma=gamma
        )
        metric_saturated_carryover = saturation_hill(
            x=m_adstocked_rw, alpha=alpha, gamma=gamma
        )
    else:
        metric_saturated_total = saturation_hill(
            x=m_adstocked_rw, alpha=alpha, gamma=gamma, x_marginal=input_total
        )
        metric_saturated_carryover = saturation_hill(
            x=m_adstocked_rw, alpha=alpha, gamma=gamma, x_marginal=input_carryover
        )

    metric_saturated_immediate = metric_saturated_total - metric_saturated_carryover

    # Decomposition
    coeff = dt_coef.loc[
        (dt_coef["solID"] == select_model) & (dt_coef["rn"] == hpm_name), "coef"
    ].iloc[0]

    m_saturated = saturation_hill(x=m_adstocked_rw, alpha=alpha, gamma=gamma)
    m_response = m_saturated * coeff
    response_total = metric_saturated_total * coeff
    response_carryover = metric_saturated_carryover * coeff
    response_immediate = response_total - response_carryover

    # Create dataframes for line and points
    dt_line = pd.DataFrame(
        {"metric": m_adstocked_rw, "response": m_response, "channel": metric_name}
    )

    if usecase == "all_historical_vec":
        dt_point = pd.DataFrame(
            {
                "input": input_total[start_rw:end_rw],
                "output": response_total,
                "ds": date_range_updated[start_rw:end_rw],
            }
        )
        dt_point_caov = pd.DataFrame(
            {"input": input_carryover[start_rw:end_rw], "output": response_carryover}
        )
        dt_point_imme = pd.DataFrame(
            {"input": input_immediate[start_rw:end_rw], "output": response_immediate}
        )
    else:
        dt_point = pd.DataFrame(
            {"input": input_total, "output": response_total, "ds": date_range_updated}
        )
        dt_point_caov = pd.DataFrame(
            {"input": input_carryover, "output": response_carryover}
        )
        dt_point_imme = pd.DataFrame(
            {"input": input_immediate, "output": response_immediate}
        )

    # Return dictionary with all results
    return {
        "metric_name": metric_name,
        "date": date_range_updated,
        "input_total": input_total,
        "input_carryover": input_carryover,
        "input_immediate": input_immediate,
        "response_total": response_total,
        "response_carryover": response_carryover,
        "response_immediate": response_immediate,
        "usecase": usecase,
        "plot": None,  # Plot will be handled separately
        "dt_line": dt_line,
        "dt_point": dt_point,
        "dt_point_caov": dt_point_caov,
        "dt_point_imme": dt_point_imme,
    }


def transform_adstock(
    x: np.ndarray,
    adstock: str,
    theta: Optional[float] = None,
    shape: Optional[float] = None,
    scale: Optional[float] = None,
    windlen: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Transform values using adstock.

    Args:
        x: Input values
        adstock: Type of adstock transformation
        theta: Theta parameter for geometric adstock
        shape: Shape parameter for Weibull adstock
        scale: Scale parameter for Weibull adstock
        windlen: Window length (defaults to length of x)

    Returns:
        Dictionary containing transformed values
    """
    # Validate adstock type
    check_adstock(adstock)

    # Set default windlen if not provided
    if windlen is None:
        windlen = len(x)

    # Apply appropriate transformation
    if adstock == "geometric":
        x_list_sim = adstock_geometric(x=x, theta=theta)
    elif adstock == "weibull_cdf":
        x_list_sim = adstock_weibull(
            x=x, shape=shape, scale=scale, windlen=windlen, type="cdf"
        )
    elif adstock == "weibull_pdf":
        x_list_sim = adstock_weibull(
            x=x, shape=shape, scale=scale, windlen=windlen, type="pdf"
        )

    return x_list_sim


def adstock_geometric(
    x: np.ndarray, theta: float
) -> Dict[str, Union[np.ndarray, float]]:
    """Apply geometric adstock transformation.

    Geometric adstock is a one-parametric adstock function with a fixed decay rate.
    For example, if TV spend on day 1 is 100€ and theta = 0.7, then:
    - day 2 has 100 x 0.7 = 70€ worth of effect carried-over from day 1
    - day 3 has 70 x 0.7 = 49€ from day 2, etc.

    Rule-of-thumb for common media genre:
    - TV: theta between 0.3 and 0.8
    - OOH/Print/Radio: theta between 0.1 and 0.4
    - Digital: theta between 0 and 0.3

    Args:
        x: Input values
        theta: Fixed decay rate parameter

    Returns:
        Dictionary containing:
        - x: Original input values
        - x_decayed: Transformed values with decay
        - theta_vec_cum: Cumulative theta values
        - inflation_total: Total inflation factor

    Raises:
        ValueError: If theta is not a single value
    """
    # Validate theta
    if not isinstance(theta, (int, float)) or np.array(theta).size != 1:
        raise ValueError("theta must be a single numeric value")

    if len(x) > 1:
        # Initialize decayed values
        x_decayed = np.zeros_like(x)
        x_decayed[0] = x[0]

        # Calculate decay
        for xi in range(1, len(x_decayed)):
            x_decayed[xi] = x[xi] + theta * x_decayed[xi - 1]

        # Calculate cumulative theta values
        theta_vec_cum = np.zeros(len(x))
        theta_vec_cum[0] = theta
        for t in range(1, len(x)):
            theta_vec_cum[t] = theta_vec_cum[t - 1] * theta
    else:
        x_decayed = x
        theta_vec_cum = np.array([theta])

    # Calculate total inflation
    inflation_total = x_decayed.sum() / x.sum()

    return {
        "x": x,
        "x_decayed": x_decayed,
        "theta_vec_cum": theta_vec_cum,
        "inflation_total": inflation_total,
    }


def normalize(x: np.ndarray) -> np.ndarray:
    """Normalize array to [0,1] range.

    Args:
        x: Input array

    Returns:
        Normalized array. If all values are the same, returns [1, 0, 0, ...]
    """
    if np.ptp(x) == 0:  # ptp is peak-to-peak, equivalent to diff(range(x))
        result = np.zeros_like(x)
        result[0] = 1
        return result
    else:
        return (x - np.min(x)) / (np.max(x) - np.min(x))


def adstock_weibull(
    x: np.ndarray,
    shape: float,
    scale: float,
    windlen: Optional[int] = None,
    type: str = "cdf",
) -> Dict[str, Union[np.ndarray, float]]:
    """Apply Weibull adstock transformation.

    Args:
        x: Input values
        shape: Shape parameter for Weibull distribution
        scale: Scale parameter for Weibull distribution
        windlen: Window length (defaults to length of x)
        type: Type of transformation ('cdf' or 'pdf')

    Returns:
        Dictionary containing transformed values and parameters
    """
    # Validate inputs
    if not isinstance(shape, (int, float)) or np.array(shape).size != 1:
        raise ValueError("shape must be a single numeric value")
    if not isinstance(scale, (int, float)) or np.array(scale).size != 1:
        raise ValueError("scale must be a single numeric value")

    # Set default windlen
    if windlen is None:
        windlen = len(x)

    if len(x) > 1:

        # Generate bins and transform scale
        x_bin = np.arange(1, windlen + 1)
        scale_trans = int(np.quantile(np.arange(1, windlen + 1), scale))

        if shape == 0 or scale == 0:
            x_decayed = x
            theta_vec_cum = theta_vec = np.zeros(windlen)
            x_imme = x
        else:
            if type.lower() == "cdf":
                # Calculate theta vector for CDF
                theta_vec = np.ones(windlen)
                theta_vec[1:] = 1 - weibull_min.cdf(
                    x_bin[:-1], shape, loc=0, scale=scale_trans
                )
                theta_vec_cum = np.cumprod(theta_vec)
            else:  # pdf
                # Calculate theta vector for PDF
                theta_vec_cum = normalize(
                    weibull_min.pdf(x_bin, shape, loc=0, scale=scale_trans)
                )

            # Calculate decayed values
            x_decayed = np.zeros((windlen, len(x)))
            for i, (x_val, x_pos) in enumerate(zip(x, range(1, len(x) + 1))):
                x_vec = np.concatenate(
                    [np.zeros(x_pos - 1), np.repeat(x_val, windlen - x_pos + 1)]
                )
                theta_vec_cum_lag = np.concatenate(
                    [np.zeros(x_pos - 1), theta_vec_cum[: (windlen - x_pos + 1)]]
                )
                x_decayed[:, i] = x_vec * theta_vec_cum_lag

            x_imme = np.diag(x_decayed)
            x_decayed = x_decayed.sum(axis=0)
    else:
        x_decayed = x_imme = x
        theta_vec_cum = np.array([1])

    # Calculate total inflation
    inflation_total = x_decayed.sum() / x.sum() if x.sum() != 0 else 1

    return {
        "x": x,
        "x_decayed": x_decayed,
        "theta_vec_cum": theta_vec_cum,
        "inflation_total": inflation_total,
        "x_imme": x_imme,
    }


def saturation_hill(
    x: np.ndarray, alpha: float, gamma: float, x_marginal: Optional[np.ndarray] = None
) -> np.ndarray:
    """Calculate Hill saturation transformation.

    Args:
        x: Input values
        alpha: Alpha parameter
        gamma: Gamma parameter
        x_marginal: Optional marginal values

    Returns:
        Transformed values

    Raises:
        ValueError: If alpha or gamma are not single values
    """
    # Validate inputs
    if not isinstance(alpha, (int, float)) or np.array(alpha).size != 1:
        raise ValueError("alpha must be a single numeric value")
    if not isinstance(gamma, (int, float)) or np.array(gamma).size != 1:
        raise ValueError("gamma must be a single numeric value")

    # Calculate inflexion point using linear interpolation
    x_range = np.array([np.min(x), np.max(x)])
    gamma_weights = np.array([1 - gamma, gamma])
    inflexion = np.dot(x_range, gamma_weights)

    # Apply Hill transformation
    if x_marginal is None:
        x_scurve = (x**alpha) / (x**alpha + inflexion**alpha)
    else:
        x_scurve = (x_marginal**alpha) / (x_marginal**alpha + inflexion**alpha)

    return x_scurve
