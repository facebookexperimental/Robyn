import numpy as np
import pandas as pd
from fbprophet import Prophet
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from robyn.modeling.entities.feature_engineering_data import FeatureEngineeringInputData, FeatureEngineeringOutputData
from typing import Tuple, Dict, Any

class FeatureEngineering:
    def __init__(self, input_data: FeatureEngineeringInputData):
        """
        Initialize the FeatureEngineering class with input data.

        Args:
            input_data (FeatureEngineeringInputData): The input data for feature engineering.
        """
        self.input_data = input_data

    def feature_engineering(self, quiet: bool = False) -> FeatureEngineeringOutputData:
        """
        Perform feature engineering on the input data.

        Args:
            quiet (bool): If True, suppress print statements. Default is False.

        Returns:
            FeatureEngineeringOutputData: The output data after feature engineering.
        """
        if not quiet:
            print(">> Running Robyn feature engineering...")

        dt_transform = self.__prepare_data()
        dt_transform_roll_wind = self.__create_rolling_window_data(dt_transform)
        dt_input_roll_wind = dt_transform.loc[
            self.input_data.rollingWindowStartWhich
            - 1 : self.input_data.rollingWindowEndWhich
            - 1
        ]
        media_cost_factor = self.__calculate_media_cost_factor(dt_input_roll_wind)

        mod_nls_collect, plot_nls_collect, yhat_collect = self.__run_models(
            dt_input_roll_wind, media_cost_factor
        )

        output_data = FeatureEngineeringOutputData(
            dt_mod=dt_transform,
            dt_modRollWind=dt_transform_roll_wind,
            dt_inputRollWind=dt_input_roll_wind,
            modNLS={
                "results": mod_nls_collect,
                "yhat": yhat_collect,
                "plots": plot_nls_collect,
            }
        )

        return output_data

    def __prepare_data(self) -> pd.DataFrame:
        """
        Prepare the input data by converting date columns and setting dependent variables.

        Returns:
            pd.DataFrame: The prepared data.
        """
        dt_input = self.input_data.dt_input.copy()
        dt_input['ds'] = pd.to_datetime(dt_input[self.input_data.date_var])
        dt_input['dep_var'] = dt_input[self.input_data.dep_var]
        return dt_input

    def __create_rolling_window_data(self, dt_transform: pd.DataFrame) -> pd.DataFrame:
        """
        Create a rolling window of data.

        Args:
            dt_transform (pd.DataFrame): The transformed data.

        Returns:
            pd.DataFrame: The rolling window data.
        """
        return dt_transform.loc[
            self.input_data.rollingWindowStartWhich
            - 1 : self.input_data.rollingWindowEndWhich
            - 1
        ]

    def __calculate_media_cost_factor(self, dt_input_roll_wind: pd.DataFrame) -> pd.Series:
        """
        Calculate the media cost factor.

        Args:
            dt_input_roll_wind (pd.DataFrame): The input data within the rolling window.

        Returns:
            pd.Series: The media cost factor.
        """
        return dt_input_roll_wind[self.input_data.paid_media_spends].sum() / dt_input_roll_wind[self.input_data.paid_media_vars].sum()

    def __run_models(self, dt_input_roll_wind: pd.DataFrame, media_cost_factor: pd.Series) -> Tuple[list, list, list]:
        """
        Run nonlinear and linear models for each media variable.

        Args:
            dt_input_roll_wind (pd.DataFrame): The input data within the rolling window.
            media_cost_factor (pd.Series): The media cost factor.

        Returns:
            tuple: Collections of model results, plots, and predictions.
        """
        mod_nls_collect = []
        plot_nls_collect = []
        yhat_collect = []

        for spend, var in zip(self.input_data.paid_media_spends, self.input_data.paid_media_vars):
            if spend != var:
                dt_spend_mod_input = dt_input_roll_wind[[spend, var]].dropna()
                results = self.__fit_spend_exposure(dt_spend_mod_input, media_cost_factor[var])
                mod_nls_collect.append(results['res'])
                plot_nls_collect.append(results['plot'])
                yhat_collect.append(results['yhat'])

        return mod_nls_collect, plot_nls_collect, yhat_collect

    def __fit_spend_exposure(self, dt_spend_mod_input: pd.DataFrame, media_cost_factor: float) -> Dict[str, Any]:
        """
        Fit the Michaelis-Menten model and a linear model to the spend and exposure data.

        Args:
            dt_spend_mod_input (pd.DataFrame): The spend and exposure data.
            media_cost_factor (float): The media cost factor.

        Returns:
            dict: The results of the model fitting, including plots and predictions.
        """
        def michaelis_menten(spend: np.ndarray, Vmax: float, Km: float) -> np.ndarray:
            return Vmax * spend / (Km + spend)

        spend = dt_spend_mod_input.iloc[:, 0]
        exposure = dt_spend_mod_input.iloc[:, 1]

        try:
            popt, _ = curve_fit(michaelis_menten, spend, exposure, maxfev=10000)
            yhat_nls = michaelis_menten(spend, *popt)
            rsq_nls = r2_score(exposure, yhat_nls)
        except:
            popt = [np.nan, np.nan]
            yhat_nls = np.full_like(spend, np.nan)
            rsq_nls = np.nan

        lm_coef = np.polyfit(spend, exposure, 1)[0]
        yhat_lm = np.polyval([lm_coef, 0], spend)
        rsq_lm = r2_score(exposure, yhat_lm)

        res = {
            'channel': dt_spend_mod_input.columns[1],
            'Vmax': popt[0],
            'Km': popt[1],
            'rsq_nls': rsq_nls,
            'rsq_lm': rsq_lm,
            'coef_lm': lm_coef
        }

        plot_data = pd.DataFrame({
            'spend': spend,
            'exposure': exposure,
            'yhat_nls': yhat_nls,
            'yhat_lm': yhat_lm
        })

        return {'res': res, 'plot': plot_data, 'yhat': yhat_nls}