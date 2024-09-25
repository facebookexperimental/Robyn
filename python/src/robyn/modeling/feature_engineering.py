from typing import List, Optional, Dict, Any, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# from prophet import Prophet

from robyn.data.entities.enums import (
    DependentVarType,
    AdstockType,
    SaturationType,
    ProphetVariableType,
    PaidMediaSigns,
    OrganicSigns,
    ContextSigns,
    ProphetSigns,
    CalibrationScope,
)

from robyn.data.entities.calibration_input import CalibrationInput, ChannelCalibrationData
from robyn.data.entities.hyperparameters import Hyperparameters, ChannelHyperparameters
from robyn.data.entities.mmmdata import MMMData


@dataclass
class FeaturizedMMMData:
    dt_mod: pd.DataFrame
    dt_modRollWind: pd.DataFrame
    modNLS: Dict[str, Any]


class FeatureEngineering:
    def __init__(self, mmm_data: MMMData, hyperparameters: Hyperparameters):
        self.mmm_data = mmm_data
        self.hyperparameters = hyperparameters

    def perform_feature_engineering(self, quiet: bool = False) -> FeaturizedMMMData:
        dt_mod = self._prepare_data()
        dt_modRollWind = self._create_rolling_window_data(dt_mod)
        media_cost_factor = self._calculate_media_cost_factor(dt_modRollWind)
        modNLS = self._run_models(dt_modRollWind, media_cost_factor)

        if "trend" in self.mmm_data.mmmdata_spec.prophet_vars:
            pass
            # dt_mod = self._prophet_decomposition(dt_mod)

        if not quiet:
            print("Feature engineering complete.")

        return FeaturizedMMMData(dt_mod=dt_mod, dt_modRollWind=dt_modRollWind, modNLS=modNLS)

    def _prepare_data(self) -> pd.DataFrame:
        dt_mod = self.mmm_data.data.copy()
        dt_mod["ds"] = pd.to_datetime(dt_mod[self.mmm_data.mmmdata_spec.date_var])
        dt_mod["dep_var"] = dt_mod[self.mmm_data.mmmdata_spec.dep_var]
        return dt_mod

    def _create_rolling_window_data(self, dt_transform: pd.DataFrame) -> pd.DataFrame:
        window_start = self.mmm_data.mmmdata_spec.window_start
        window_end = self.mmm_data.mmmdata_spec.window_end

        if window_start is None and window_end is None:
            # If both are None, return the entire DataFrame
            return dt_transform
        elif window_start is None:
            # If only start is None, filter up to end
            return dt_transform[dt_transform["ds"] <= window_end]
        elif window_end is None:
            # If only end is None, filter from start
            return dt_transform[dt_transform["ds"] >= window_start]
        else:
            # If both are provided, filter between start and end
            return dt_transform[(dt_transform["ds"] >= window_start) & (dt_transform["ds"] <= window_end)]

    def _calculate_media_cost_factor(self, dt_input_roll_wind: pd.DataFrame) -> pd.Series:
        total_spend = dt_input_roll_wind[self.mmm_data.mmmdata_spec.paid_media_spends].sum().sum()
        return dt_input_roll_wind[self.mmm_data.mmmdata_spec.paid_media_spends].sum() / total_spend

    def _run_models(self, dt_modRollWind: pd.DataFrame, media_cost_factor: float) -> Dict[str, Dict[str, Any]]:
        modNLS = {}
        for paid_media_var in self.mmm_data.mmmdata_spec.paid_media_spends:
            result = self._fit_spend_exposure(dt_modRollWind, paid_media_var, media_cost_factor)
            if result is not None:
                modNLS[paid_media_var] = result

        # Keep the plot windows open
        plt.show()

        return modNLS

    def _fit_spend_exposure(
        self, dt_modRollWind: pd.DataFrame, paid_media_var: str, media_cost_factor: float
    ) -> Dict[str, Any]:
        """
        Fit the Michaelis-Menten model and a linear model to the spend and exposure data.

        Args:
            dt_modRollWind (pd.DataFrame): The rolling window data.
            paid_media_var (str): The name of the paid media variable.
            media_cost_factor (float): The media cost factor.

        Returns:
            dict: The results of the model fitting, including plots and predictions.
        """
        print(f"Processing {paid_media_var}")

        def michaelis_menten(x, Vmax, Km):
            return Vmax * x / (Km + x)  # our version at the moment is x = spend, y is exposure.

        # new version would be flipped.

        def linear_model(x, a, b):
            return a * x + b

        spend_var = paid_media_var
        exposure_var = self.mmm_data.mmmdata_spec.paid_media_vars[
            self.mmm_data.mmmdata_spec.paid_media_spends.index(paid_media_var)
        ]

        if spend_var not in dt_modRollWind.columns or exposure_var not in dt_modRollWind.columns:
            print(f"Error: {spend_var} or {exposure_var} not found in dt_modRollWind")
            return None

        spend_data = dt_modRollWind[spend_var]
        exposure_data = dt_modRollWind[exposure_var]

        if spend_data.empty or exposure_data.empty:
            print(f"Error: spend_data or exposure_data is empty for {paid_media_var}")
            return None

        print(f"spend_data range: {spend_data.min()} - {spend_data.max()}")
        print(f"exposure_data range: {exposure_data.min()} - {exposure_data.max()}")

        try:
            # Fit Michaelis-Menten model
            popt_nls, _ = curve_fit(
                michaelis_menten,
                spend_data,
                exposure_data,
                p0=[max(exposure_data), np.median(spend_data)],
                bounds=([0, 0], [np.inf, np.inf]),
            )

            # Fit linear model
            popt_lm, _ = curve_fit(linear_model, spend_data, exposure_data)

            # Calculate R-squared for both models
            yhat_nls = michaelis_menten(spend_data, *popt_nls)
            yhat_lm = linear_model(spend_data, *popt_lm)
            rsq_nls = 1 - np.sum((exposure_data - yhat_nls) ** 2) / np.sum(
                (exposure_data - np.mean(exposure_data)) ** 2
            )
            rsq_lm = 1 - np.sum((exposure_data - yhat_lm) ** 2) / np.sum((exposure_data - np.mean(exposure_data)) ** 2)

            # Create plot
            plt.figure(figsize=(10, 6))
            plt.scatter(spend_data, exposure_data, label="Data")
            spend_range = np.linspace(min(spend_data), max(spend_data), 100)
            plt.plot(spend_range, michaelis_menten(spend_range, *popt_nls), label="NLS fit")
            plt.plot(spend_range, linear_model(spend_range, *popt_lm), label="Linear fit")
            plt.xlabel(f"Spend [{spend_var}]")
            plt.ylabel(f"Exposure [{exposure_var}]")
            plt.title(f"Spend vs Exposure for {paid_media_var}")
            plt.legend()
            plt.show(block=False)

            res = {
                "channel": paid_media_var,
                "Vmax": popt_nls[0],
                "Km": popt_nls[1],
                "rsq_nls": rsq_nls,
                "rsq_lm": rsq_lm,
                "coef_lm": popt_lm[0],
            }

            plot_data = pd.DataFrame(
                {"spend": spend_data, "exposure": exposure_data, "yhat_nls": yhat_nls, "yhat_lm": yhat_lm}
            )

            return {"res": res, "plot": plot_data, "yhat": yhat_nls}

        except Exception as e:
            print(f"Error fitting models for {paid_media_var}: {str(e)}")
            return None

    @staticmethod
    def _hill_function(x, alpha, gamma):
        return x**alpha / (x**alpha + gamma**alpha)

    def _prophet_decomposition(self, dt_transform: pd.DataFrame) -> pd.DataFrame:
        from prophet import Prophet

        model = Prophet(holidays=self._set_holidays())
        model.fit(dt_transform[["ds", "dep_var"]].rename(columns={"dep_var": "y"}))

        future = model.make_future_dataframe(periods=0)
        forecast = model.predict(future)

        dt_transform["trend"] = forecast["trend"]
        dt_transform["season"] = forecast["seasonal"]

        return dt_transform

    def _set_holidays(self) -> pd.DataFrame:
        if self.mmm_data.mmmdata_spec.prophet_country:
            from prophet.holidays import get_holiday_names, load_holidays

            country_holidays = get_holiday_names(self.mmm_data.mmmdata_spec.prophet_country)
            holidays = load_holidays(
                self.mmm_data.mmmdata_spec.prophet_country,
                years=[self.mmm_data.data["ds"].dt.year.min(), self.mmm_data.data["ds"].dt.year.max()],
            )

            return holidays[holidays["holiday"].isin(country_holidays)]
        else:
            return pd.DataFrame()  # Return empty DataFrame if no country specified

    def _apply_transformations(self, x: pd.Series, params: ChannelHyperparameters) -> pd.Series:
        x_adstock = self._apply_adstock(x, params)
        x_saturated = self._apply_saturation(x_adstock, params)
        return x_saturated

    def _apply_adstock(self, x: pd.Series, params: ChannelHyperparameters) -> pd.Series:
        if self.hyperparameters.adstock == AdstockType.GEOMETRIC:
            return self._geometric_adstock(x, params.thetas[0])
        elif self.hyperparameters.adstock in [AdstockType.WEIBULL_CDF, AdstockType.WEIBULL_PDF]:
            return self._weibull_adstock(x, params.shapes[0], params.scales[0])
        else:
            raise ValueError(f"Unsupported adstock type: {self.hyperparameters.adstock}")

    @staticmethod
    def _geometric_adstock(x: pd.Series, theta: float) -> pd.Series:
        return x.ewm(alpha=1 - theta, adjust=False).mean()

    @staticmethod
    def _weibull_adstock(x: pd.Series, shape: float, scale: float) -> pd.Series:
        def weibull_pdf(t):
            return (shape / scale) * ((t / scale) ** (shape - 1)) * np.exp(-((t / scale) ** shape))

        weights = [weibull_pdf(t) for t in range(1, len(x) + 1)]
        weights = weights / np.sum(weights)
        return np.convolve(x, weights[::-1], mode="full")[: len(x)]

    @staticmethod
    def _apply_saturation(x: pd.Series, params: ChannelHyperparameters) -> pd.Series:
        alpha, gamma = params.alphas[0], params.gammas[0]
        return x**alpha / (x**alpha + gamma**alpha)
