# pyre-strict

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass
from robyn.data.entities.mmmdata import MMMData
from robyn.modeling.entities.modeloutputs import ModelOutputs


@dataclass
class HillParameters:
    """
    Represents the parameters of the Hill function for a specific channel.

    Attributes:
        alpha (float): The alpha parameter of the Hill function.
        gamma (float): The gamma parameter of the Hill function.
    """

    alpha: float
    gamma: float


@dataclass
class ResponseCurveData:
    """
    Represents the response curve data for a specific model and metric.

    Attributes:
        model_id (str): The ID of the model.
        metric_name (str): The name of the metric.
        channel (str): The name of the channel.
        spend_values (np.ndarray): Array of spend values.
        response_values (np.ndarray): Array of response values corresponding to spend values.
        hill_params (HillParameters): The Hill function parameters used for the calculation.
    """

    model_id: str
    metric_name: str
    channel: str
    spend_values: np.ndarray
    response_values: np.ndarray
    hill_params: HillParameters


class ResponseCurveCalculator:
    """
    Calculates response curves for marketing mix models.

    This class handles the calculation of response curves for different models and metrics,
    including the retrieval of Hill function parameters and the calculation of response values.

    Attributes:
        mmm_data (MMMData): Input data for the marketing mix model.
        model_outputs (ModelOutputs): Output data from the model runs.
    """

    def __init__(self, mmm_data: MMMData, model_outputs: ModelOutputs):
        """
        Initialize the ResponseCurveCalculator.

        Args:
            mmm_data (MMMData): Input data for the marketing mix model.
            model_outputs (ModelOutputs): Output data from the model runs.
        """
        self.mmm_data = mmm_data
        self.model_outputs = model_outputs
        
    def calculate_response(self, 
                           select_model: str,
                           metric_name: str,
                           metric_value: Optional[Union[float, List[float]]] = None,
                           date_range: Optional[str] = None,
                           quiet: bool = False,
                           json_file: Optional[str] = None,
                           robyn_object: Optional[str] = None) -> Dict[str, Any]:
        """
        Calculate the response for a given model and metric.

        Args:
            select_model (str): The ID of the model to use for calculations.
            metric_name (str): The name of the metric to calculate the response for.
            metric_value (Optional[Union[float, List[float]]]): A specific metric value or list of values to use.
            date_range (Optional[str]): The date range to consider for calculations.
            quiet (bool): Whether to suppress informational messages.
            json_file (Optional[str]): Path to a JSON file containing model data.
            robyn_object (Optional[str]): Path to a Robyn object file containing model data.

        Returns:
            Dict[str, Any]: A dictionary containing the calculated response data.
        """
        # Get input data
        dt_input = self.mmm_data.data
        # if json_file:
        #     # Load data from JSON file
        #     # Update self.mmm_data and self.model_outputs accordingly
        # elif robyn_object:
        #     # Load data from Robyn object
        #     # Update self.mmm_data and self.model_outputs accordingly

        dt_hyppar = self._get_hyperparam_data()
        dt_coef = self._get_coefficient_data()
        
        # Check inputs and determine use case
        usecase = self._which_usecase(metric_value, date_range)
        metric_type = self._check_metric_type(metric_name)
        all_dates = dt_input[self.mmm_data.mmmdata_spec.date_var]
        all_values = dt_input[metric_name]

        # Process dates and values
        ds_list = self._check_metric_dates(date_range, all_dates, quiet)
        val_list = self._check_metric_value(metric_value, metric_name, all_values, ds_list['metric_loc'])
        date_range_updated = ds_list['date_range_updated']
        metric_value_updated = val_list['metric_value_updated']
        all_values_updated = val_list['all_values_updated']

        # Transform exposure to spend if necessary
        if metric_type == "exposure":
            all_values_updated = self._transform_exposure_to_spend(metric_name, metric_value_updated, all_values_updated, ds_list['metric_loc'])
            hpm_name = self._get_spend_name(metric_name)
        else:
            hpm_name = metric_name

        # Apply adstock transformation
        adstock_params = self._get_adstock_params(select_model, hpm_name, dt_hyppar)
        media_vec_origin = dt_input[metric_name]
        x_list = self._transform_adstock(media_vec_origin, adstock_params)
        m_adstocked = x_list['x_decayed']

        x_list_sim = self._transform_adstock(all_values_updated, adstock_params)
        media_vec_sim = x_list_sim['x_decayed']
        media_vec_sim_imme = x_list_sim['x_imme'] if adstock_params['type'] == "weibull_pdf" else x_list_sim['x']
        
        input_total = media_vec_sim[ds_list['metric_loc']]
        input_immediate = media_vec_sim_imme[ds_list['metric_loc']]
        input_carryover = input_total - input_immediate

        # Apply saturation
        alpha, gamma = self._get_saturation_params(select_model, hpm_name, dt_hyppar)
        m_adstockedRW = m_adstocked[self.mmm_data.mmmdata_spec.window_start:self.mmm_data.mmmdata_spec.window_end]
        
        if usecase == "all_historical_vec":
            metric_saturated_total = self._saturation_hill(m_adstockedRW, alpha, gamma)
            metric_saturated_carryover = self._saturation_hill(m_adstockedRW, alpha, gamma)
        else:
            metric_saturated_total = self._saturation_hill(m_adstockedRW, alpha, gamma, x_marginal=input_total)
            metric_saturated_carryover = self._saturation_hill(m_adstockedRW, alpha, gamma, x_marginal=input_carryover)
        
        metric_saturated_immediate = metric_saturated_total - metric_saturated_carryover

        # Calculate response
        coeff = dt_coef[(dt_coef['solID'] == select_model) & (dt_coef['rn'] == hpm_name)]['coef'].values[0]
        m_saturated = self._saturation_hill(m_adstockedRW, alpha, gamma)
        m_response = m_saturated * coeff
        response_total = metric_saturated_total * coeff
        response_carryover = metric_saturated_carryover * coeff
        response_immediate = response_total - response_carryover

        # Prepare output
        output = {
            "metric_name": metric_name,
            "date": date_range_updated,
            "input_total": input_total,
            "input_carryover": input_carryover,
            "input_immediate": input_immediate,
            "response_total": response_total,
            "response_carryover": response_carryover,
            "response_immediate": response_immediate,
            "usecase": usecase,
            "plot": self._create_response_plot(m_adstockedRW, m_response, input_total, response_total, 
                                               input_carryover, response_carryover, input_immediate, 
                                               response_immediate, metric_name, metric_type, date_range_updated)
        }
        
        return output

    def _which_usecase(self, metric_value: Optional[Union[float, List[float]]], date_range: Optional[str]) -> str:
        if metric_value is None and date_range is None:
            return "all_historical_vec"
        elif metric_value is None and date_range is not None:
            return "selected_historical_vec"
        elif isinstance(metric_value, (int, float)) and date_range is None:
            return "total_metric_default_range"
        elif isinstance(metric_value, (int, float)) and date_range is not None:
            return "total_metric_selected_range"
        elif isinstance(metric_value, (list, np.ndarray)) and date_range is None:
            return "unit_metric_default_last_n"
        else:
            return "unit_metric_selected_dates"

    def _check_metric_type(self, metric_name: str) -> str:
        if metric_name in self.mmm_data.mmmdata_spec.paid_media_spends:
            return "spend"
        elif metric_name in self.mmm_data.mmmdata_spec.paid_media_vars:
            return "exposure"
        elif metric_name in self.mmm_data.mmmdata_spec.organic_vars:
            return "organic"
        else:
            raise ValueError(f"Unknown metric type for {metric_name}")

    def _check_metric_dates(self, date_range: Optional[str], all_dates: pd.Series, quiet: bool) -> Dict[str, Any]:
        start_rw = self.mmm_data.mmmdata_spec.window_start
        end_rw = self.mmm_data.mmmdata_spec.window_end

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

        return {
            "metric_loc": metric_loc,
            "date_range_updated": date_range_updated
        }

    def _check_metric_value(self, metric_value: Optional[Union[float, List[float]]], 
                            metric_name: str, all_values: pd.Series, metric_loc: slice) -> Dict[str, Any]:
        if metric_value is None:
            metric_value_updated = all_values[metric_loc]
        elif isinstance(metric_value, (int, float)):
            metric_value_updated = np.full(len(all_values[metric_loc]), metric_value)
        else:
            metric_value_updated = np.array(metric_value)

        all_values_updated = all_values.copy()
        all_values_updated[metric_loc] = metric_value_updated

        return {
            "metric_value_updated": metric_value_updated,
            "all_values_updated": all_values_updated
        }

    def _transform_exposure_to_spend(self, metric_name: str, metric_value_updated: np.ndarray, 
                                     all_values_updated: pd.Series, metric_loc: slice) -> pd.Series:
        spend_name = self._get_spend_name(metric_name)
        spend_expo_mod = self.mmm_data.mmmdata_spec.modNLS['results']
        temp = spend_expo_mod[spend_expo_mod['channel'] == metric_name]
        
        if temp['rsq_nls'].values[0] > temp['rsq_lm'].values[0]:
            Vmax = temp['Vmax'].values[0]
            Km = temp['Km'].values[0]
            input_immediate = self._mic_men(metric_value_updated, Vmax, Km, reverse=True)
        else:
            coef_lm = temp['coef_lm'].values[0]
            input_immediate = metric_value_updated / coef_lm

        all_values_updated[metric_loc] = input_immediate
        return all_values_updated

    def _get_spend_name(self, metric_name: str) -> str:
        return self.mmm_data.mmmdata_spec.paid_media_spends[
            self.mmm_data.mmmdata_spec.paid_media_vars.index(metric_name)]

    def _mic_men(self, x: np.ndarray, Vmax: float, Km: float, reverse: bool = False) -> np.ndarray:
        if reverse:
            return Km * x / (Vmax - x)
        else:
            return Vmax * x / (Km + x)

    def _get_adstock_params(self, select_model: str, hpm_name: str, dt_hyppar: pd.DataFrame) -> Dict[str, Any]:
        adstock_type = self.model_outputs.hyper_fixed['adstock']
        params = {'type': adstock_type}
        
        if adstock_type == "geometric":
            params['theta'] = dt_hyppar[(dt_hyppar['solID'] == select_model) & 
                                        (dt_hyppar['name'] == f"{hpm_name}_thetas")]['value'].values[0]
        elif "weibull" in adstock_type:
            params['shape'] = dt_hyppar[(dt_hyppar['solID'] == select_model) & 
                                        (dt_hyppar['name'] == f"{hpm_name}_shapes")]['value'].values[0]
            params['scale'] = dt_hyppar[(dt_hyppar['solID'] == select_model) & 
                                        (dt_hyppar['name'] == f"{hpm_name}_scales")]['value'].values[0]
        
        return params

    def _transform_adstock(self, x: np.ndarray, params: Dict[str, Any]) -> Dict[str, np.ndarray]:
        if params['type'] == "geometric":
            x_decayed = self._geometric_adstock(x, params['theta'])
            return {'x': x, 'x_decayed': x_decayed}
        elif params['type'] == "weibull_cdf":
            x_decayed = self._weibull_adstock(x, params['shape'], params['scale'], cumulative=True)
            return {'x': x, 'x_decayed': x_decayed}
        elif params['type'] == "weibull_pdf":
            x_decayed, x_imme = self._weibull_adstock(x, params['shape'], params['scale'], cumulative=False)
            return {'x': x, 'x_decayed': x_decayed, 'x_imme': x_imme}

    def _geometric_adstock(self, x: np.ndarray, theta: float) -> np.ndarray:
        return np.convolve(x, theta**np.arange(len(x)))[:len(x)]

    def _weibull_adstock(self, x: np.ndarray, shape: float, scale: float, cumulative: bool) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
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

    def _get_saturation_params(self, select_model: str, hpm_name: str, dt_hyppar: pd.DataFrame) -> Tuple[float, float]:
        alpha = dt_hyppar[(dt_hyppar['solID'] == select_model) & 
                          (dt_hyppar['name'] == f"{hpm_name}_alphas")]['value'].values[0]
        gamma = dt_hyppar[(dt_hyppar['solID'] == select_model) & 
                          (dt_hyppar['name'] == f"{hpm_name}_gammas")]['value'].values[0]
        return alpha, gamma

    def _saturation_hill(self, x: np.ndarray, alpha: float, gamma: float, x_marginal: Optional[np.ndarray] = None) -> np.ndarray:
        if x_marginal is None:
            return x**gamma / (x**gamma + alpha**gamma)
        else:
            return (x + x_marginal)**gamma / ((x + x_marginal)**gamma + alpha**gamma) - x**gamma / (x**gamma + alpha**gamma)

    def _create_response_plot(self, m_adstockedRW: np.ndarray, m_response: np.ndarray, 
                              input_total: np.ndarray, response_total: np.ndarray,
                              input_carryover: np.ndarray, response_carryover: np.ndarray,
                              input_immediate: np.ndarray, response_immediate: np.ndarray,
                              metric_name: str, metric_type: str, date_range_updated: pd.Series) -> plt.Figure:
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
    
    def get_hyperparam_data(self) -> pd.DataFrame:
        return pd.concat([trial.result_hyp_param for trial in self.model_outputs.trials])
    def get_coefficient_data(self) -> pd.DataFrame:
        return pd.concat([trial.x_decomp_agg for trial in self.model_outputs.trials])
