# pyre-strict

from typing import Optional, Union, Literal
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from robyn.data.entities.mmmdata import MMMData
from robyn.modeling.entities.modeloutputs import ModelOutputs
from robyn.data.entities.enums import AdstockType
from robyn.data.entities.hyperparameters import ChannelHyperparameters, Hyperparameters

@dataclass
class MetricDateInfo:
    metric_loc: pd.Series
    date_range_updated: pd.Series

@dataclass
class MetricValueInfo:
    metric_value_updated: np.ndarray
    all_values_updated: pd.Series

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

@dataclass
class AdstockOutput:
    x: np.ndarray
    x_decayed: np.ndarray
    x_imme: Optional[np.ndarray] = None

class UseCase(str, Enum):
    ALL_HISTORICAL_VEC = "all_historical_vec"
    SELECTED_HISTORICAL_VEC = "selected_historical_vec"
    TOTAL_METRIC_DEFAULT_RANGE = "total_metric_default_range"
    TOTAL_METRIC_SELECTED_RANGE = "total_metric_selected_range"
    UNIT_METRIC_DEFAULT_LAST_N = "unit_metric_default_last_n"
    UNIT_METRIC_SELECTED_DATES = "unit_metric_selected_dates"

class ResponseCurveCalculator:
    def __init__(self, mmm_data: MMMData, model_outputs: ModelOutputs, hyperparameter: Hyperparameters):
        self.mmm_data: MMMData = mmm_data
        self.model_outputs: ModelOutputs = model_outputs
        self.hyperparameter: Hyperparameters = hyperparameter

    def calculate_response(self, 
                           select_model: str,
                           metric_name: str,
                           metric_value: Optional[Union[float, list[float]]] = None,
                           date_range: Optional[str] = None,
                           quiet: bool = False,
                           dt_hyppar: pd.DataFrame = pd.DataFrame(),
                           dt_coef: pd.DataFrame = pd.DataFrame()
                        ) -> ResponseOutput:
        # Determine the use case based on input parameters
        usecase = self._which_usecase(metric_value, date_range)
        
        # Check the metric type (spend, exposure, organic)
        metric_type = self._check_metric_type(metric_name)
        
        all_dates = self.mmm_data.data[self.mmm_data.mmmdata_spec.date_var]
        all_values = self.mmm_data.data[metric_name]
        
        # Check and process date range
        ds_list = self._check_metric_dates(date_range, all_dates, quiet)
        
        # Check and process metric values
        val_list = self._check_metric_value(metric_value, metric_name, all_values, ds_list.metric_loc)
        
        date_range_updated = ds_list.date_range_updated
        metric_value_updated = val_list.metric_value_updated
        all_values_updated = val_list.all_values_updated

        # Transform exposure to spend if necessary
        if metric_type == "exposure":
            all_values_updated = self._transform_exposure_to_spend(metric_name, metric_value_updated, all_values_updated, ds_list.metric_loc)
            hpm_name = self._get_spend_name(metric_name)
        else:
            hpm_name = metric_name

        media_vec_origin = self.mmm_data.data[metric_name]

        # Get adstock parameters and apply adstock transformation
        adstock_params = self._get_adstock_params(select_model, hpm_name, dt_hyppar)
        x_list = self._transform_adstock(media_vec_origin, adstock_params)
        m_adstocked = x_list.x_decayed

        x_list_sim = self._transform_adstock(all_values_updated, adstock_params)
        media_vec_sim = x_list_sim.x_decayed
        
        input_total = media_vec_sim[ds_list.metric_loc]
        if self.hyperparameter.adstock == AdstockType.WEIBULL_PDF:
            media_vec_sim_imme = x_list_sim.x_imme
            input_immediate = media_vec_sim_imme[ds_list.metric_loc]
        else:
            input_immediate = x_list_sim.x[ds_list.metric_loc]
        input_carryover = input_total - input_immediate

        # Get saturation parameters and apply saturation
        hill_params = self._get_saturation_params(select_model, hpm_name, dt_hyppar)
        
        m_adstockedRW = m_adstocked[self.mmm_data.mmmdata_spec.rolling_window_start_which:self.mmm_data.mmmdata_spec.rolling_window_end_which]
        
        if usecase == UseCase.ALL_HISTORICAL_VEC:
            metric_saturated_total = self._saturation_hill(m_adstockedRW, hill_params)
            metric_saturated_carryover = self._saturation_hill(m_adstockedRW, hill_params)
        else:
            metric_saturated_total = self._saturation_hill(m_adstockedRW, hill_params, x_marginal=input_total)
            metric_saturated_carryover = self._saturation_hill(m_adstockedRW, hill_params, x_marginal=input_carryover)
        
        metric_saturated_immediate = metric_saturated_total - metric_saturated_carryover

        # Calculate final response values
        coeff = dt_coef[(dt_coef['solID'] == select_model) & (dt_coef['rn'] == hpm_name)]['coef'].values[0]
        m_saturated = self._saturation_hill(m_adstockedRW, hill_params)
        m_response = m_saturated * coeff
        response_total = metric_saturated_total * coeff
        response_carryover = metric_saturated_carryover * coeff
        response_immediate = response_total - response_carryover

        # Create response plot
        plot = self._create_response_plot(m_adstockedRW, m_response, input_total, response_total, 
                                          input_carryover, response_carryover, input_immediate, 
                                          response_immediate, metric_name, metric_type, date_range_updated)

        return ResponseOutput(
            metric_name=metric_name,
            date=date_range_updated,
            input_total=input_total,
            input_carryover=input_carryover,
            input_immediate=input_immediate,
            response_total=response_total,
            response_carryover=response_carryover,
            response_immediate=response_immediate,
            usecase=usecase,
            plot=plot
        )


    def _which_usecase(self, metric_value: Optional[Union[float, list[float]]], date_range: Optional[str]) -> UseCase:
        if metric_value is None:
            return UseCase.ALL_HISTORICAL_VEC if date_range is None or date_range == "all" else UseCase.SELECTED_HISTORICAL_VEC
        elif isinstance(metric_value, (int, float)):
            return UseCase.TOTAL_METRIC_DEFAULT_RANGE if date_range is None else UseCase.TOTAL_METRIC_SELECTED_RANGE
        elif isinstance(metric_value, (list, np.ndarray)):
            return UseCase.UNIT_METRIC_DEFAULT_LAST_N if date_range is None else UseCase.UNIT_METRIC_SELECTED_DATES
        else:
            raise ValueError(f"Invalid metric_value type: {type(metric_value)}")

    def _check_metric_type(self, metric_name: str) -> Literal["spend", "exposure", "organic"]:
        if metric_name in self.mmm_data.mmmdata_spec.paid_media_spends:
            return "spend"
        elif metric_name in self.mmm_data.mmmdata_spec.paid_media_vars:
            return "exposure"
        elif metric_name in self.mmm_data.mmmdata_spec.organic_vars:
            return "organic"
        else:
            raise ValueError(f"Unknown metric type for {metric_name}")

    def _check_metric_dates(self, date_range: Optional[str], all_dates: pd.Series, quiet: bool) -> MetricDateInfo:
        start_rw = self.mmm_data.mmmdata_spec.rolling_window_start_which
        end_rw = self.mmm_data.mmmdata_spec.rolling_window_end_which

        if date_range == "all" or date_range is None:
            metric_loc = slice(start_rw, end_rw)
            date_range_updated = all_dates[metric_loc]
        elif date_range.startswith("last_"):
            n_periods = int(date_range.split("_")[1])
            metric_loc = slice(end_rw - n_periods + 1, end_rw)
            date_range_updated = all_dates[metric_loc]
        elif isinstance(date_range, list) and len(date_range) == 2:
            start_date, end_date = pd.to_datetime(date_range)
            metric_loc = (all_dates >= start_date) & (all_dates <= end_date)
            date_range_updated = all_dates[metric_loc]
        else:
            raise ValueError(f"Invalid date_range: {date_range}")

        if not quiet:
            print(f"Using date range: {date_range_updated.iloc[0]} to {date_range_updated.iloc[-1]}")

        return MetricDateInfo(metric_loc=metric_loc, date_range_updated=date_range_updated)

    def _check_metric_value(self, metric_value: Optional[Union[float, list[float]]], 
                            metric_name: str, all_values: pd.Series, metric_loc: Union[slice, pd.Series]) -> MetricValueInfo:
        # if isinstance(metric_loc, slice):
        #     selected_values = all_values.iloc[metric_loc]
        # elif isinstance(metric_loc, pd.Series):
        #     selected_values = all_values[metric_loc]
        # else:  # metric_loc is a list of dates
        #     selected_values = all_values[all_values.index.isin(metric_loc)]

        if metric_value is None:
            metric_value_updated = all_values[metric_loc]
        elif isinstance(metric_value, (int, float)):
            metric_value_updated = np.full(len(all_values[metric_loc]), metric_value)
        else:
            metric_value_updated = np.array(metric_value)
            if len(metric_value_updated) != len(all_values[metric_loc]):
                raise ValueError(f"Length of metric_value ({len(metric_value_updated)}) does not match the selected date range ({len(all_values[metric_loc])})")

        all_values_updated = all_values.copy()
        all_values_updated[metric_loc] = metric_value_updated

        return MetricValueInfo(metric_value_updated=metric_value_updated, all_values_updated=all_values_updated)

    def _transform_exposure_to_spend(self, metric_name: str, metric_value_updated: np.ndarray, 
                                     all_values_updated: pd.Series, metric_loc: Union[slice, pd.Series]) -> pd.Series:
        spend_name = self._get_spend_name(metric_name)
        spend_expo_mod = self.mmm_data.mmmdata_spec.modNLS['results']
        temp = spend_expo_mod[spend_expo_mod['channel'] == metric_name]
        
        if temp['rsq_nls'].values[0] > temp['rsq_lm'].values[0]:
            # Use non-linear least squares model
            Vmax = temp['Vmax'].values[0]
            Km = temp['Km'].values[0]
            input_immediate = Km * metric_value_updated / (Vmax - metric_value_updated)
        else:
            # Use linear model
            coef_lm = temp['coef_lm'].values[0]
            input_immediate = metric_value_updated / coef_lm

        all_values_updated[metric_loc] = input_immediate
        return all_values_updated

    def _get_spend_name(self, metric_name: str) -> str:
        return self.mmm_data.mmmdata_spec.paid_media_spends[
            self.mmm_data.mmmdata_spec.paid_media_vars.index(metric_name)]

    def _get_adstock_params(self, select_model: str, hpm_name: str, dt_hyppar: pd.DataFrame) -> ChannelHyperparameters:
        adstock_type = self.hyperparameter.adstock
        params = ChannelHyperparameters()
        
        if adstock_type == AdstockType.GEOMETRIC:
            params.thetas = dt_hyppar[dt_hyppar['solID'] == select_model][f"{hpm_name}_thetas"].values[0]
        elif adstock_type in [AdstockType.WEIBULL, AdstockType.WEIBULL_CDF, AdstockType.WEIBULL_PDF]:
            params.shapes = dt_hyppar[dt_hyppar['solID'] == select_model][f"{hpm_name}_shapes"].values[0]
            params.scales = dt_hyppar[dt_hyppar['solID'] == select_model][f"{hpm_name}_scales"].values[0]
        
        return params

    # TODO: Move this to transform.py
    def _transform_adstock(self, x: np.ndarray, params: ChannelHyperparameters) -> AdstockOutput:
        adstock_type = self.hyperparameter.adstock
        if adstock_type == AdstockType.GEOMETRIC:
            x_decayed = self._geometric_adstock(x, params.thetas)
            return AdstockOutput(x=x, x_decayed=x_decayed)
        elif adstock_type == AdstockType.WEIBULL_CDF:
            x_decayed = self._weibull_adstock(x, params.shapes, params.scales, cumulative=True)
            return AdstockOutput(x=x, x_decayed=x_decayed)
        elif adstock_type == AdstockType.WEIBULL_PDF:
            x_decayed, x_imme = self._weibull_adstock(x, params.shapes, params.scales, cumulative=False)
            return AdstockOutput(x=x, x_decayed=x_decayed, x_imme=x_imme)

    def _geometric_adstock(self, x: np.ndarray, theta: float) -> np.ndarray:
        return np.convolve(x, theta**np.arange(len(x)))[:len(x)]

    def _weibull_adstock(self, x: np.ndarray, shape: float, scale: float, cumulative: bool) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        n = len(x)
        weights = np.array([scale * (((i + 1) / scale) ** (shape - 1)) * np.exp(-((i + 1) / scale) ** shape) for i in range(n)])
        
        if cumulative:
            weights = 1 - np.exp(-((np.arange(1, n+1) / scale) ** shape))
        
        x_decayed = np.convolve(x, weights)[:n]
        
        if cumulative:
            return x_decayed
        else:
            x_imme = x * weights[0]
            return x_decayed, x_imme
    
    def _get_saturation_params(self, select_model: str, hpm_name: str, dt_hyppar: pd.DataFrame) -> ChannelHyperparameters:
        params = ChannelHyperparameters()
        params.alphas = dt_hyppar[dt_hyppar['solID'] == select_model][f"{hpm_name}_alphas"].values[0]
        params.gammas = dt_hyppar[dt_hyppar['solID'] == select_model][f"{hpm_name}_gammas"].values[0]
        return params

    def _saturation_hill(self, x: np.ndarray, hill_params: ChannelHyperparameters, x_marginal: Optional[np.ndarray] = None) -> np.ndarray:
        alpha, gamma = hill_params.alphas, hill_params.gammas
        if x_marginal is None:
            return x**gamma / (x**gamma + alpha**gamma)
        else:
            return (x + x_marginal)**gamma / ((x + x_marginal)**gamma + alpha**gamma) - x**gamma / (x**gamma + alpha**gamma)

    def _create_response_plot(self, m_adstockedRW: np.ndarray, m_response: np.ndarray, 
                              input_total: np.ndarray, response_total: np.ndarray,
                              input_carryover: np.ndarray, response_carryover: np.ndarray,
                              input_immediate: np.ndarray, response_immediate: np.ndarray,
                              metric_name: str, metric_type: Literal["spend", "exposure", "organic"], 
                              date_range_updated: pd.Series) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(10, 6))
    
        ax.plot(m_adstockedRW, m_response, color='steelblue', label='Response curve')
        ax.scatter(input_total, response_total, color='red', s=50, label='Total response')
        
        if len(np.unique(input_total)) == 1:
            ax.scatter(input_carryover, response_carryover, color='green', s=50, marker='s', label='Carryover response')
            ax.scatter(input_immediate, response_immediate, color='orange', s=50, marker='^', label='Immediate response')

        ax.set_xlabel('Input')
        ax.set_ylabel('Response')
        ax.set_title(f"Saturation curve of {'organic' if metric_type == 'organic' else 'paid'} media: {metric_name}")
        ax.legend()
        
        if len(np.unique(input_total)) == 1:
            subtitle = (f"Carryover Response: {response_carryover[0]:.2f} @ Input {input_carryover[0]:.2f}\n"
                        f"Immediate Response: {response_immediate[0]:.2f} @ Input {input_immediate[0]:.2f}\n"
                        f"Total (C+I) Response: {response_total[0]:.2f} @ Input {input_total[0]:.2f}")
            ax.text(0.05, 0.95, subtitle, transform=ax.transAxes, verticalalignment='top', fontsize=9)
        
        plt.figtext(0.5, 0.01, 
                    f"Response period: {date_range_updated.iloc[0]} to {date_range_updated.iloc[-1]} [{len(date_range_updated)} periods]", 
                    ha="center", fontsize=8)
        
        return fig