# pyre-strict

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Literal, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from robyn.data.entities.enums import AdstockType
from robyn.data.entities.hyperparameters import ChannelHyperparameters, Hyperparameters
from robyn.data.entities.mmmdata import MMMData
from robyn.modeling.entities.modeloutputs import ModelOutputs
from robyn.modeling.transformations.transformations import Transformation

# Initialize logger
logger = logging.getLogger(__name__)

@dataclass
class MetricDateInfo:
    metric_loc: pd.Series
    date_range_updated: pd.Series

    def __str__(self) -> str:
        return f"MetricDateInfo(date_range: {self.date_range_updated.iloc[0]} to {self.date_range_updated.iloc[-1]})"

@dataclass
class MetricValueInfo:
    metric_value_updated: np.ndarray
    all_values_updated: pd.Series

    def __str__(self) -> str:
        return f"MetricValueInfo(values_shape: {self.metric_value_updated.shape})"

@dataclass
class ResponseOutput:
    metric_name: str
    date: pd.Series
    input_total: np.ndarray
    input_carryover: np.ndarray
    input_immediate: np.ndarray
    response_total: np.ndarray
    response_carryover: np.ndarray
    response_immediate: np.ndarray
    usecase: str
    plot: plt.Figure

    def __str__(self) -> str:
        return (
            f"ResponseOutput(metric: {self.metric_name}, "
            f"date_range: {self.date.iloc[0]} to {self.date.iloc[-1]}, "
            f"usecase: {self.usecase})"
        )

class UseCase(str, Enum):
    ALL_HISTORICAL_VEC = "all_historical_vec"
    SELECTED_HISTORICAL_VEC = "selected_historical_vec"
    TOTAL_METRIC_DEFAULT_RANGE = "total_metric_default_range"
    TOTAL_METRIC_SELECTED_RANGE = "total_metric_selected_range"
    UNIT_METRIC_DEFAULT_LAST_N = "unit_metric_default_last_n"
    UNIT_METRIC_SELECTED_DATES = "unit_metric_selected_dates"

class ResponseCurveCalculator:
    def __init__(
        self,
        mmm_data: MMMData,
        model_outputs: ModelOutputs,
        hyperparameter: Hyperparameters,
    ):
        logger.debug("Initializing ResponseCurveCalculator with data: %s", mmm_data)
        self.mmm_data: MMMData = mmm_data
        self.model_outputs: ModelOutputs = model_outputs
        self.hyperparameter: Hyperparameters = hyperparameter
        self.transformation = Transformation(mmm_data)
        logger.info("ResponseCurveCalculator initialized successfully")

    def calculate_response(
        self,
        select_model: str,
        metric_name: str,
        metric_value: Optional[Union[float, list[float]]] = None,
        date_range: Optional[str] = None,
        quiet: bool = False,
        dt_hyppar: pd.DataFrame = pd.DataFrame(),
        dt_coef: pd.DataFrame = pd.DataFrame(),
    ) -> ResponseOutput:
        logger.info("Starting response calculation for metric: %s", metric_name)
        logger.debug("Input parameters - model: %s, date_range: %s", select_model, date_range)

        # Determine the use case based on input parameters
        usecase = self._which_usecase(metric_value, date_range)
        logger.debug("Determined use case: %s", usecase)

        # Check the metric type
        metric_type = self._check_metric_type(metric_name)
        logger.debug("Metric type: %s", metric_type)

        all_dates = self.mmm_data.data[self.mmm_data.mmmdata_spec.date_var]
        all_values = self.mmm_data.data[metric_name]

        # Process date range and metric values
        try:
            ds_list = self._check_metric_dates(date_range, all_dates, quiet)
            logger.debug("Date range processed: %s", ds_list)

            val_list = self._check_metric_value(
                metric_value, metric_name, all_values, ds_list.metric_loc
            )
            logger.debug("Metric values processed: %s", val_list)
        except ValueError as e:
            logger.error("Error processing dates or values: %s", e)
            raise

        date_range_updated = ds_list.date_range_updated
        metric_value_updated = val_list.metric_value_updated
        all_values_updated = val_list.all_values_updated

        # Transform exposure to spend if necessary
        if metric_type == "exposure":
            logger.debug("Transforming exposure to spend for metric: %s", metric_name)
            all_values_updated = self._transform_exposure_to_spend(
                metric_name,
                metric_value_updated,
                all_values_updated,
                ds_list.metric_loc,
            )
            hpm_name = self._get_spend_name(metric_name)
        else:
            hpm_name = metric_name

        logger.debug("Processing media vector for metric: %s", metric_name)
        media_vec_origin = self.mmm_data.data[metric_name]

        # Apply adstock transformation
        adstockType = self.hyperparameter.adstock
        channel_hyperparams = self._get_channel_hyperparams(
            select_model, hpm_name, dt_hyppar
        )
        logger.debug("Applying adstock transformation with type: %s", adstockType)
        
        try:
            x_list = self.transformation.transform_adstock(
                media_vec_origin, adstockType, channel_hyperparams
            )
            x_list_sim = self.transformation.transform_adstock(
                all_values_updated, adstockType, channel_hyperparams
            )
        except Exception as e:
            logger.error("Error in adstock transformation: %s", e)
            raise

        media_vec_sim = x_list_sim.x_decayed
        input_total = media_vec_sim[ds_list.metric_loc]
        
        if self.hyperparameter.adstock == AdstockType.WEIBULL_PDF:
            logger.debug("Processing Weibull PDF adstock")
            media_vec_sim_imme = x_list_sim.x_imme
            input_immediate = media_vec_sim_imme[ds_list.metric_loc]
        else:
            input_immediate = x_list_sim.x[ds_list.metric_loc]
        input_carryover = input_total - input_immediate

        # Apply saturation transformation
        logger.debug("Applying saturation transformation")
        hill_params = self._get_saturation_params(select_model, hpm_name, dt_hyppar)

        m_adstockedRW = x_list.x_decayed[
            self.mmm_data.mmmdata_spec.rolling_window_start_which : self.mmm_data.mmmdata_spec.rolling_window_end_which
        ]

        try:
            if usecase == UseCase.ALL_HISTORICAL_VEC:
                logger.debug("Processing historical vector saturation")
                metric_saturated_total = self.transformation.saturation_hill(
                    m_adstockedRW, hill_params.alphas[0], hill_params.gammas[0]
                )
                metric_saturated_carryover = self.transformation.saturation_hill(
                    m_adstockedRW, hill_params.alphas[0], hill_params.gammas[0]
                )
            else:
                metric_saturated_total = self.transformation.saturation_hill(
                    m_adstockedRW,
                    hill_params.alphas[0],
                    hill_params.gammas[0],
                    x_marginal=input_total,
                )
                metric_saturated_carryover = self.transformation.saturation_hill(
                    m_adstockedRW,
                    hill_params.alphas[0],
                    hill_params.gammas[0],
                    x_marginal=input_carryover,
                )
        except Exception as e:
            logger.error("Error in saturation transformation: %s", e)
            raise

        metric_saturated_immediate = metric_saturated_total - metric_saturated_carryover

        # Calculate final response values
        logger.debug("Calculating final response values")
        try:
            coeff = dt_coef[
                (dt_coef["solID"] == select_model) & (dt_coef["rn"] == hpm_name)
            ]["coef"].values[0]
            
            m_saturated = self.transformation.saturation_hill(
                m_adstockedRW, hill_params.alphas[0], hill_params.gammas[0]
            )
            m_response = m_saturated * coeff
            response_total = metric_saturated_total * coeff
            response_carryover = metric_saturated_carryover * coeff
            response_immediate = response_total - response_carryover
        except Exception as e:
            logger.error("Error calculating final response values: %s", e)
            raise

        response_output = ResponseOutput(
            metric_name=metric_name,
            date=date_range_updated,
            input_total=input_total,
            input_carryover=input_carryover,
            input_immediate=input_immediate,
            response_total=response_total,
            response_carryover=response_carryover,
            response_immediate=response_immediate,
            usecase=usecase,
            plot=None,
        )
        
        logger.info("Response calculation completed successfully: %s", response_output)
        return response_output

    def _which_usecase(
        self,
        metric_value: Optional[Union[float, list[float]]],
        date_range: Optional[str],
    ) -> UseCase:
        logger.debug("Determining use case - metric_value type: %s, date_range: %s", 
                    type(metric_value) if metric_value is not None else None, 
                    date_range)
        
        try:
            if metric_value is None:
                usecase = (
                    UseCase.ALL_HISTORICAL_VEC
                    if date_range is None or date_range == "all"
                    else UseCase.SELECTED_HISTORICAL_VEC
                )
            elif isinstance(metric_value, (int, float)):
                usecase = (
                    UseCase.TOTAL_METRIC_DEFAULT_RANGE
                    if date_range is None
                    else UseCase.TOTAL_METRIC_SELECTED_RANGE
                )
            elif isinstance(metric_value, (list, np.ndarray)):
                usecase = (
                    UseCase.UNIT_METRIC_DEFAULT_LAST_N
                    if date_range is None
                    else UseCase.UNIT_METRIC_SELECTED_DATES
                )
            else:
                raise ValueError(f"Invalid metric_value type: {type(metric_value)}")
            
            logger.debug("Use case determined: %s", usecase)
            return usecase
        except Exception as e:
            logger.error("Error determining use case: %s", e)
            raise

    def _check_metric_type(
        self, metric_name: str
    ) -> Literal["spend", "exposure", "organic"]:
        logger.debug("Checking metric type for: %s", metric_name)
        
        try:
            if metric_name in self.mmm_data.mmmdata_spec.paid_media_spends:
                metric_type = "spend"
            elif metric_name in self.mmm_data.mmmdata_spec.paid_media_vars:
                metric_type = "exposure"
            elif metric_name in self.mmm_data.mmmdata_spec.organic_vars:
                metric_type = "organic"
            else:
                logger.error("Unknown metric type for: %s", metric_name)
                raise ValueError(f"Unknown metric type for {metric_name}")
            
            logger.debug("Metric type determined: %s", metric_type)
            return metric_type
        except Exception as e:
            logger.error("Error checking metric type: %s", e)
            raise

# [Previous code remains the same until _check_metric_type...]

    def _check_metric_dates(
        self, date_range: Optional[str], all_dates: pd.Series, quiet: bool
    ) -> MetricDateInfo:
        logger.debug("Checking metric dates - date_range: %s", date_range)
        
        try:
            start_rw = self.mmm_data.mmmdata_spec.rolling_window_start_which
            end_rw = self.mmm_data.mmmdata_spec.rolling_window_end_which
            
            if date_range == "all" or date_range is None:
                logger.debug("Using full date range")
                metric_loc = slice(start_rw, end_rw)
                date_range_updated = all_dates[metric_loc]
            elif isinstance(date_range, str) and date_range.startswith("last_"):
                n_periods = int(date_range.split("_")[1])
                logger.debug("Using last %d periods", n_periods)
                metric_loc = slice(end_rw - n_periods, end_rw)
                date_range_updated = all_dates[metric_loc]
            elif isinstance(date_range, list) and len(date_range) == 2:
                start_date, end_date = pd.to_datetime(date_range)
                logger.debug("Using custom date range: %s to %s", start_date, end_date)
                metric_loc = (all_dates >= start_date) & (all_dates <= end_date)
                date_range_updated = all_dates[metric_loc]
            else:
                logger.error("Invalid date_range format: %s", date_range)
                raise ValueError(f"Invalid date_range: {date_range}")

            if not quiet:
                logger.info("Date range selected: %s to %s", 
                           date_range_updated.iloc[0], 
                           date_range_updated.iloc[-1])

            result = MetricDateInfo(metric_loc=metric_loc, date_range_updated=date_range_updated)
            logger.debug("Metric dates processed: %s", result)
            return result
            
        except Exception as e:
            logger.error("Error processing metric dates: %s", e)
            raise

    def _check_metric_value(
        self,
        metric_value: Optional[Union[float, list[float]]],
        metric_name: str,
        all_values: pd.Series,
        metric_loc: Union[slice, pd.Series],
    ) -> MetricValueInfo:
        logger.debug("Checking metric value for %s - value type: %s", 
                    metric_name, 
                    type(metric_value) if metric_value is not None else None)
        
        try:
            if metric_value is None:
                logger.debug("Using existing values from data")
                metric_value_updated = all_values[metric_loc]
            elif isinstance(metric_value, (int, float)):
                logger.debug("Using constant value: %f", metric_value)
                metric_value_updated = np.full(len(all_values[metric_loc]), metric_value)
            else:
                logger.debug("Using provided value array of length %d", len(metric_value))
                metric_value_updated = np.array(metric_value)
                if len(metric_value_updated) != len(all_values[metric_loc]):
                    logger.error(
                        "Length mismatch: metric_value length=%d, expected length=%d",
                        len(metric_value_updated),
                        len(all_values[metric_loc])
                    )
                    raise ValueError(
                        f"Length of metric_value ({len(metric_value_updated)}) does not match "
                        f"the selected date range ({len(all_values[metric_loc])})"
                    )

            all_values_updated = all_values.copy()
            all_values_updated[metric_loc] = metric_value_updated

            result = MetricValueInfo(
                metric_value_updated=metric_value_updated,
                all_values_updated=all_values_updated
            )
            logger.debug("Metric value processing complete: %s", result)
            return result
            
        except Exception as e:
            logger.error("Error processing metric value: %s", e)
            raise

    def _transform_exposure_to_spend(
        self,
        metric_name: str,
        metric_value_updated: np.ndarray,
        all_values_updated: pd.Series,
        metric_loc: Union[slice, pd.Series],
    ) -> pd.Series:
        logger.debug("Transforming exposure to spend for metric: %s", metric_name)
        
        try:
            spend_name = self._get_spend_name(metric_name)
            spend_expo_mod = self.mmm_data.mmmdata_spec.modNLS["results"]
            temp = spend_expo_mod[spend_expo_mod["channel"] == metric_name]
            
            logger.debug("Found model parameters - rsq_nls: %f, rsq_lm: %f", 
                        temp["rsq_nls"].values[0], 
                        temp["rsq_lm"].values[0])

            if temp["rsq_nls"].values[0] > temp["rsq_lm"].values[0]:
                logger.debug("Using non-linear least squares model")
                Vmax = temp["Vmax"].values[0]
                Km = temp["Km"].values[0]
                input_immediate = Km * metric_value_updated / (Vmax - metric_value_updated)
            else:
                logger.debug("Using linear model")
                coef_lm = temp["coef_lm"].values[0]
                input_immediate = metric_value_updated / coef_lm

            all_values_updated[metric_loc] = input_immediate
            logger.debug("Exposure to spend transformation complete")
            return all_values_updated
            
        except Exception as e:
            logger.error("Error transforming exposure to spend: %s", e)
            raise

    def _get_spend_name(self, metric_name: str) -> str:
        logger.debug("Getting spend name for metric: %s", metric_name)
        try:
            spend_name = self.mmm_data.mmmdata_spec.paid_media_spends[
                self.mmm_data.mmmdata_spec.paid_media_vars.index(metric_name)
            ]
            logger.debug("Found spend name: %s", spend_name)
            return spend_name
        except Exception as e:
            logger.error("Error getting spend name: %s", e)
            raise

    def _get_channel_hyperparams(
        self, select_model: str, hpm_name: str, dt_hyppar: pd.DataFrame
    ) -> ChannelHyperparameters:
        logger.debug("Getting channel hyperparameters for model: %s, metric: %s", 
                    select_model, hpm_name)
        
        try:
            adstock_type = self.hyperparameter.adstock
            params = ChannelHyperparameters()

            if adstock_type == AdstockType.GEOMETRIC:
                logger.debug("Using geometric adstock type")
                params.thetas = dt_hyppar[dt_hyppar["solID"] == select_model][
                    f"{hpm_name}_thetas"
                ].values
            elif adstock_type in [
                AdstockType.WEIBULL,
                AdstockType.WEIBULL_CDF,
                AdstockType.WEIBULL_PDF,
            ]:
                logger.debug("Using Weibull adstock type: %s", adstock_type)
                params.shapes = dt_hyppar[dt_hyppar["solID"] == select_model][
                    f"{hpm_name}_shapes"
                ].values
                params.scales = dt_hyppar[dt_hyppar["solID"] == select_model][
                    f"{hpm_name}_scales"
                ].values

            logger.debug("Channel hyperparameters retrieved successfully")
            return params
            
        except Exception as e:
            logger.error("Error getting channel hyperparameters: %s", e)
            raise

    def _get_saturation_params(
        self, select_model: str, hpm_name: str, dt_hyppar: pd.DataFrame
    ) -> ChannelHyperparameters:
        logger.debug("Getting saturation parameters for model: %s, metric: %s", 
                    select_model, hpm_name)
        
        try:
            params = ChannelHyperparameters()
            params.alphas = dt_hyppar[dt_hyppar["solID"] == select_model][
                f"{hpm_name}_alphas"
            ].values
            params.gammas = dt_hyppar[dt_hyppar["solID"] == select_model][
                f"{hpm_name}_gammas"
            ].values
            
            logger.debug("Saturation parameters - alphas: %s, gammas: %s", 
                        params.alphas, params.gammas)
            return params
            
        except Exception as e:
            logger.error("Error getting saturation parameters: %s", e)
            raise

    def _create_response_plot(
        self,
        m_adstockedRW: np.ndarray,
        m_response: np.ndarray,
        input_total: np.ndarray,
        response_total: np.ndarray,
        input_carryover: np.ndarray,
        response_carryover: np.ndarray,
        input_immediate: np.ndarray,
        response_immediate: np.ndarray,
        metric_name: str,
        metric_type: Literal["spend", "exposure", "organic"],
        date_range_updated: pd.Series,
    ) -> plt.Figure:
        logger.debug("Creating response plot for metric: %s", metric_name)
        
        try:
            fig, ax = plt.subplots(figsize=(10, 6))

            ax.plot(m_adstockedRW, m_response, color="steelblue", label="Response curve")
            ax.scatter(
                input_total, response_total, color="red", s=50, label="Total response"
            )

            if len(np.unique(input_total)) == 1:
                logger.debug("Adding single-point response details to plot")
                ax.scatter(
                    input_carryover,
                    response_carryover,
                    color="green",
                    s=50,
                    marker="s",
                    label="Carryover response",
                )
                ax.scatter(
                    input_immediate,
                    response_immediate,
                    color="orange",
                    s=50,
                    marker="^",
                    label="Immediate response",
                )

            ax.set_xlabel("Input")
            ax.set_ylabel("Response")
            ax.set_title(
                f"Saturation curve of {'organic' if metric_type == 'organic' else 'paid'} media: {metric_name}"
            )
            ax.legend()

            if len(np.unique(input_total)) == 1:
                subtitle = (
                    f"Carryover Response: {response_carryover[0]:.2f} @ Input {input_carryover[0]:.2f}\n"
                    f"Immediate Response: {response_immediate[0]:.2f} @ Input {input_immediate[0]:.2f}\n"
                    f"Total (C+I) Response: {response_total[0]:.2f} @ Input {input_total[0]:.2f}"
                )
                ax.text(
                    0.05,
                    0.95,
                    subtitle,
                    transform=ax.transAxes,
                    verticalalignment="top",
                    fontsize=9,
                )

            plt.figtext(
                0.5,
                0.01,
                f"Response period: {date_range_updated.iloc[0]} to {date_range_updated.iloc[-1]} [{len(date_range_updated)} periods]",
                ha="center",
                fontsize=8,
            )

            logger.debug("Response plot created successfully")
            return fig
            
        except Exception as e:
            logger.error("Error creating response plot: %s", e)
            raise