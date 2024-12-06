# pyre-strict

from typing import Optional, Dict, Any
import logging
import pandas as pd
import warnings
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
import numpy as np

np.float_ = np.float64
from prophet import Prophet
from robyn.data.entities.holidays_data import HolidaysData

from robyn.data.entities.enums import (
    AdstockType,
)
from robyn.modeling.entities.featurized_mmm_data import FeaturizedMMMData
from robyn.data.entities.hyperparameters import Hyperparameters, ChannelHyperparameters
from robyn.data.entities.mmmdata import MMMData


logger = logging.getLogger(__name__)


class FeatureEngineering:
    """
    A class used to perform feature engineering for Marketing Mix Modeling (MMM) data.
    """

    def __init__(
        self,
        mmm_data: MMMData,
        hyperparameters: Hyperparameters,
        holidays_data: Optional[HolidaysData] = None,
    ):
        self.mmm_data = mmm_data
        self.hyperparameters = hyperparameters
        self.holidays_data = holidays_data
        self.logger = logger
        self.logger.debug(
            "Initializing FeatureEngineering with MMM data and hyperparameters"
        )

    def perform_feature_engineering(self, quiet: bool = False) -> FeaturizedMMMData:
        self.logger.info("Starting feature engineering process")
        self.logger.debug(f"Input data shape: {self.mmm_data.data.shape}")

        dt_transform = self._prepare_data()
        self.logger.debug(f"Prepared data shape: {dt_transform.shape}")

        if any(
            var in self.holidays_data.prophet_vars
            for var in ["trend", "season", "holiday", "monthly", "weekday"]
        ):
            self.logger.info("Starting Prophet decomposition")
            dt_transform = self._prophet_decomposition(dt_transform)
            if not quiet:
                self.logger.info("Prophet decomposition complete")

        # Include all independent variables
        all_ind_vars = (
            self.holidays_data.prophet_vars
            + self.mmm_data.mmmdata_spec.context_vars
            + self.mmm_data.mmmdata_spec.paid_media_spends
            + self.mmm_data.mmmdata_spec.organic_vars
        )
        self.logger.debug(f"Processing {len(all_ind_vars)} independent variables")

        dt_mod = dt_transform
        dt_modRollWind = self._create_rolling_window_data(dt_transform)
        self.logger.debug(f"Rolling window data shape: {dt_modRollWind.shape}")

        media_cost_factor = self._calculate_media_cost_factor(dt_modRollWind)
        self.logger.debug("Calculated media cost factors")

        modNLS = self._run_models(dt_modRollWind, media_cost_factor)
        self.logger.debug(f"Completed model runs for {len(modNLS['results'])} channels")

        columns_to_keep = ["ds", "dep_var"] + all_ind_vars
        # Only keep columns that exist in both dataframes
        columns_to_keep = [
            col
            for col in columns_to_keep
            if col in dt_mod.columns and col in dt_modRollWind.columns
        ]
        self.logger.debug(f"Keeping {len(columns_to_keep)} columns in final dataset")

        dt_mod = dt_transform[columns_to_keep]
        dt_modRollWind = dt_modRollWind[columns_to_keep]

        if not quiet:
            self.logger.info("Feature engineering complete")

        # Fill or interpolate missing values in dt_mod
        missing_before = dt_mod.isnull().sum().sum()
        dt_mod = dt_mod.fillna(method="ffill").fillna(method="bfill")
        missing_after = dt_mod.isnull().sum().sum()
        self.logger.info(f"Filled {missing_before - missing_after} missing values")

        return FeaturizedMMMData(
            dt_mod=dt_mod, dt_modRollWind=dt_modRollWind, modNLS=modNLS
        )

    def _prepare_data(self) -> pd.DataFrame:
        self.logger.debug("Starting data preparation")
        dt_transform = self.mmm_data.data.copy()
        dt_transform["ds"] = pd.to_datetime(
            dt_transform[self.mmm_data.mmmdata_spec.date_var]
        ).dt.strftime("%Y-%m-%d")
        dt_transform["dep_var"] = dt_transform[self.mmm_data.mmmdata_spec.dep_var]
        dt_transform["competitor_sales_B"] = dt_transform["competitor_sales_B"].astype(
            "int64"
        )
        self.logger.debug("Data preparation complete")
        return dt_transform

    def _create_rolling_window_data(self, dt_transform: pd.DataFrame) -> pd.DataFrame:
        self.logger.debug("Creating rolling window data")
        window_start = self.mmm_data.mmmdata_spec.window_start
        window_end = self.mmm_data.mmmdata_spec.window_end

        try:
            if window_start is None and window_end is None:
                self.logger.debug("No window constraints specified")
                return dt_transform
            elif window_start is None:
                if window_end < dt_transform["ds"].min():
                    self.logger.error("Window end date is before the start of data")
                    raise ValueError("window_end is before the start of the data")
                self.logger.debug(f"Filtering data up to {window_end}")
                return dt_transform[dt_transform["ds"] <= window_end]
            elif window_end is None:
                if window_start > dt_transform["ds"].max():
                    self.logger.error("Window start date is after the end of data")
                    raise ValueError("window_start is after the end of the data")
                self.logger.debug(f"Filtering data from {window_start}")
                return dt_transform[dt_transform["ds"] >= window_start]
            else:
                if window_start > window_end:
                    self.logger.error(
                        f"Invalid window range: {window_start} to {window_end}"
                    )
                    raise ValueError("window_start is after window_end")
                self.logger.debug(
                    f"Filtering data between {window_start} and {window_end}"
                )
                return dt_transform[
                    (dt_transform["ds"] >= window_start)
                    & (dt_transform["ds"] <= window_end)
                ]
        except Exception as e:
            self.logger.error(f"Error creating rolling window data: {str(e)}")
            raise

    def _run_models(
        self, dt_modRollWind: pd.DataFrame, media_cost_factor: pd.Series
    ) -> Dict[str, Any]:
        """Run models only for channels where exposure metrics differ from spend metrics."""
        self.logger.info(
            "Starting model runs for paid media variables with different exposure metrics"
        )
        modNLS = {"results": [], "yhat": [], "plots": {}}
        # Get indices where exposure differs from spend
        exposure_selector = [
            i
            for i, (spend, exposure) in enumerate(
                zip(
                    self.mmm_data.mmmdata_spec.paid_media_spends,
                    self.mmm_data.mmmdata_spec.paid_media_vars,
                )
            )
            if spend != exposure
        ]
        for idx in exposure_selector:
            paid_media_spend = self.mmm_data.mmmdata_spec.paid_media_spends[idx]
            paid_media_var = self.mmm_data.mmmdata_spec.paid_media_vars[idx]
            self.logger.debug(
                f"Processing model for {paid_media_var} (spend: {paid_media_spend})"
            )
            # Prepare data for modeling
            dt_spend_mod_input = pd.DataFrame(
                {
                    "spend": dt_modRollWind[paid_media_spend],
                    "exposure": dt_modRollWind[paid_media_var],
                }
            )
            result = self._fit_spend_exposure(
                dt_spend_mod_input, paid_media_var, media_cost_factor[paid_media_spend]
            )
            if result is not None:
                modNLS["results"].append(result["res"])
                # Store both NLS and LM predictions
                for model_type in ["nls", "lm"]:
                    yhat_data = result["plot"].copy()
                    yhat_data["channel"] = paid_media_var
                    yhat_data["models"] = model_type
                    yhat_data["ds"] = dt_modRollWind["ds"]  # Add date column if needed
                    yhat_data = yhat_data.rename(
                        columns={"spend": "x", "exposure": "y"}
                    )  # Rename columns to match R structure
                    modNLS["yhat"].extend(yhat_data.to_dict(orient="records"))
                modNLS["plots"][paid_media_var] = result["plot"]
                self.logger.debug(
                    f"Completed model fit for {paid_media_var} with R² = {result['res']['rsq']:.4f}"
                )
            else:
                self.logger.warning(f"Model fitting failed for {paid_media_var}")
        self.logger.info(f"Completed model runs for {len(modNLS['results'])} channels")
        return modNLS

    def _fit_spend_exposure(
        self,
        dt_spend_mod_input: pd.DataFrame,
        paid_media_var: str,
        media_cost_factor: float,
    ) -> Dict[str, Any]:
        """Fit spend-exposure models matching R implementation."""
        self.logger.info(f"Fitting spend-exposure model for {paid_media_var}")

        def michaelis_menten(x, Vmax, Km):
            return Vmax * x / (Km + x)

        spend_data = dt_spend_mod_input["spend"]
        exposure_data = dt_spend_mod_input["exposure"]
        if exposure_data.empty or spend_data.empty:
            self.logger.warning(
                f"Empty data for {paid_media_var}. Skipping model fitting."
            )
            return None
        try:
            # Initial parameters based on R implementation
            p0 = [exposure_data.max(), exposure_data.max() / 2]
            # Fit Michaelis-Menten (NLS) model
            popt_nls, _ = curve_fit(
                michaelis_menten,
                spend_data,
                exposure_data,
                p0=p0,
                bounds=([0, 0], [np.inf, np.inf]),
                maxfev=10000,
            )
            yhat_nls = michaelis_menten(spend_data, *popt_nls)
            rsq_nls = 1 - np.sum((exposure_data - yhat_nls) ** 2) / np.sum(
                (exposure_data - exposure_data.mean()) ** 2
            )
            # Calculate AIC and BIC for NLS
            aic_nls = 2 * len(popt_nls) - 2 * np.log(
                np.sum((exposure_data - yhat_nls) ** 2)
            )
            bic_nls = len(popt_nls) * np.log(len(exposure_data)) - 2 * np.log(
                np.sum((exposure_data - yhat_nls) ** 2)
            )
            # Fit linear model
            lm = LinearRegression(fit_intercept=False)
            lm.fit(spend_data.values.reshape(-1, 1), exposure_data)
            yhat_lm = lm.predict(spend_data.values.reshape(-1, 1))
            rsq_lm = 1 - np.sum((exposure_data - yhat_lm) ** 2) / np.sum(
                (exposure_data - exposure_data.mean()) ** 2
            )
            # Calculate AIC and BIC for LM
            aic_lm = 2 * 1 - 2 * np.log(np.sum((exposure_data - yhat_lm) ** 2))
            bic_lm = 1 * np.log(len(exposure_data)) - 2 * np.log(
                np.sum((exposure_data - yhat_lm) ** 2)
            )

            # Choose the better model based on R-squared
            if rsq_nls > rsq_lm:
                model_type = "nls"
                rsq = rsq_nls
                yhat = yhat_nls
            else:
                model_type = "lm"
                rsq = rsq_lm
                yhat = yhat_lm
            # Prepare result dictionary
            res = {
                "channel": paid_media_var,
                "model_type": (
                    "nls" if rsq_nls > rsq_lm else "lm"
                ),  # Set model_type based on the better model
                "Vmax": float(popt_nls[0]) if rsq_nls > rsq_lm else None,
                "Km": float(popt_nls[1]) if rsq_nls > rsq_lm else None,
                "aic_nls": aic_nls if rsq_nls > rsq_lm else None,
                "aic_lm": aic_lm,
                "bic_nls": bic_nls if rsq_nls > rsq_lm else None,
                "bic_lm": bic_lm,
                "rsq_nls": rsq_nls if rsq_nls > rsq_lm else None,
                "rsq_lm": rsq_lm,
                "coef_lm": float(lm.coef_[0]),
                "rsq": rsq,  # Add a combined rsq for easier access
            }
            plot_data = pd.DataFrame(
                {
                    "spend": spend_data.values,
                    "exposure": exposure_data.values,
                    "yhat": yhat,
                }
            )
            return {"res": res, "plot": plot_data, "yhat": yhat}
        except Exception as e:
            self.logger.warning(
                f"Error fitting models for {paid_media_var}: {str(e)}. Falling back to linear model."
            )
            try:
                lm = LinearRegression(fit_intercept=False)
                lm.fit(spend_data.values.reshape(-1, 1), exposure_data)
                yhat_lm = lm.predict(spend_data.values.reshape(-1, 1))
                rsq_lm = 1 - np.sum((exposure_data - yhat_lm) ** 2) / np.sum(
                    (exposure_data - exposure_data.mean()) ** 2
                )
                # Calculate AIC and BIC for LM
                aic_lm = 2 * 1 - 2 * np.log(np.sum((exposure_data - yhat_lm) ** 2))
                bic_lm = 1 * np.log(len(exposure_data)) - 2 * np.log(
                    np.sum((exposure_data - yhat_lm) ** 2)
                )
                res = {
                    "channel": paid_media_var,
                    "Vmax": None,  # Set to None if not applicable
                    "Km": None,  # Set to None if not applicable
                    "aic_nls": None,  # Set to None if not applicable
                    "aic_lm": aic_lm,
                    "bic_nls": None,  # Set to None if not applicable
                    "bic_lm": bic_lm,
                    "rsq_nls": None,  # Set to None if not applicable
                    "rsq_lm": rsq_lm,
                    "coef_lm": float(lm.coef_[0]),
                    "rsq": rsq_lm,  # Use rsq_lm as the combined rsq
                }
                plot_data = pd.DataFrame(
                    {
                        "spend": spend_data.values,
                        "exposure": exposure_data.values,
                        "yhat": yhat_lm,
                    }
                )
                return {"res": res, "plot": plot_data, "yhat": yhat_lm}
            except Exception as e2:
                self.logger.error(
                    f"Both NLS and linear model fitting failed for {paid_media_var}: {str(e2)}"
                )
                return None

    def _calculate_media_cost_factor(
        self, dt_input_roll_wind: pd.DataFrame
    ) -> pd.Series:
        """Calculate media cost factors matching R implementation."""
        spend_sums = dt_input_roll_wind[
            self.mmm_data.mmmdata_spec.paid_media_spends
        ].sum()
        exposure_sums = dt_input_roll_wind[
            self.mmm_data.mmmdata_spec.paid_media_vars
        ].sum()
        return spend_sums / exposure_sums

    @staticmethod
    def _hill_function(x, alpha, gamma):
        return x**alpha / (x**alpha + gamma**alpha)

    def _prophet_decomposition(self, dt_mod: pd.DataFrame) -> pd.DataFrame:
        """Modified Prophet decomposition to match R implementation"""
        self.logger.info("Starting Prophet decomposition")

        # Prepare recurrence data
        recurrence = dt_mod[["ds", "dep_var"]].rename(columns={"dep_var": "y"}).copy()
        recurrence["ds"] = pd.to_datetime(recurrence["ds"])

        # Get prophet parameters
        prophet_vars = self.holidays_data.prophet_vars
        use_trend = "trend" in prophet_vars
        use_holiday = "holiday" in prophet_vars
        use_season = "season" in prophet_vars or "yearly.seasonality" in prophet_vars
        use_monthly = "monthly" in prophet_vars
        use_weekday = "weekday" in prophet_vars or "weekly.seasonality" in prophet_vars

        # Process holidays
        holidays = self._set_holidays(
            dt_mod,
            self.holidays_data.dt_holidays.copy(),
            self.mmm_data.mmmdata_spec.interval_type,
        )

        # Prepare base regressors
        dt_regressors = pd.concat(
            [
                recurrence,
                dt_mod[
                    self.mmm_data.mmmdata_spec.paid_media_spends
                    + self.mmm_data.mmmdata_spec.context_vars
                    + self.mmm_data.mmmdata_spec.organic_vars
                ].copy(),
            ],
            axis=1,
        )
        dt_regressors["ds"] = pd.to_datetime(dt_regressors["ds"])

        # Handle factor variables
        if self.mmm_data.mmmdata_spec.factor_vars is None:
            self.mmm_data.set_default_factor_vars()
        factor_vars = self.mmm_data.mmmdata_spec.factor_vars
        if factor_vars:
            # Create dummy variables but keep original
            dt_factors = dt_mod[factor_vars].copy()
            dt_ohe = pd.get_dummies(dt_factors, prefix=factor_vars[0], prefix_sep="_")

            # Remove the reference level (usually the most frequent one)
            reference_level = dt_factors[factor_vars[0]].mode()[0]
            reference_col = f"{factor_vars[0]}_{reference_level}"
            if reference_col in dt_ohe.columns:
                dt_ohe = dt_ohe.drop(columns=[reference_col])

            # Add dummies to regressors
            dt_regressors = pd.concat([dt_regressors, dt_ohe], axis=1)

        # Setup Prophet model
        model = Prophet(
            holidays=(
                holidays[holidays["country"].isin([self.holidays_data.prophet_country])]
                if use_holiday
                else None
            ),
            yearly_seasonality=use_season,
            weekly_seasonality=(
                use_weekday if self.mmm_data.mmmdata_spec.day_interval <= 7 else False
            ),
            daily_seasonality=False,
        )

        if use_monthly:
            model.add_seasonality(name="monthly", period=30.5, fourier_order=5)

        # Add factor regressors
        if factor_vars and len(dt_ohe.columns) > 0:
            for col in dt_ohe.columns:
                model.add_regressor(col)

        # Fit model and get forecast
        model.fit(dt_regressors)
        forecast = model.predict(dt_regressors)

        # Update original dataframe with decomposition results
        these = range(len(recurrence))
        if use_trend:
            dt_mod["trend"] = forecast["trend"].iloc[these].values
        if use_season:
            dt_mod["season"] = forecast["yearly"].iloc[these].values
        if use_monthly:
            dt_mod["monthly"] = forecast["monthly"].iloc[these].values
        if use_weekday:
            dt_mod["weekday"] = forecast["weekly"].iloc[these].values
        if use_holiday:
            dt_mod["holiday"] = forecast["holidays"].iloc[these].values

        # Handle factor variables in output
        if factor_vars:
            for var in factor_vars:
                factor_cols = [
                    col for col in forecast.columns if col.startswith(f"{var}_")
                ]
                factor_effects = forecast[factor_cols].sum(axis=1)

                # Set baseline (reference level) to 0
                baseline = factor_effects.min()
                dt_mod[var] = factor_effects - baseline

                # Scale effects to match R's implementation
                if dt_mod[var].sum() > 0:
                    scale_factor = (
                        dt_mod[var].max() / 1272202.89877032
                    )  # R's maximum value
                    dt_mod[var] = dt_mod[var] / scale_factor

        return dt_mod

    def _set_holidays(
        self, dt_transform: pd.DataFrame, dt_holidays: pd.DataFrame, interval_type: str
    ) -> pd.DataFrame:
        self.logger.debug(f"Setting holidays for interval type: {interval_type}")

        try:
            dt_transform["ds"] = pd.to_datetime(dt_transform["ds"])
            holidays = dt_holidays.copy()
            holidays["ds"] = pd.to_datetime(holidays["ds"])

            if interval_type == "day":
                self.logger.debug("Using daily holiday data")
                return holidays
            elif interval_type == "week":
                self.logger.debug("Processing weekly holiday aggregation")
                week_start = dt_transform["ds"].dt.weekday[0]
                holidays["ds"] = (
                    holidays["ds"]
                    - pd.to_timedelta(holidays["ds"].dt.weekday, unit="D")
                    + pd.Timedelta(days=week_start)
                )
                holidays = (
                    holidays.groupby(["ds", "country", "year"])
                    .agg(
                        holiday=("holiday", lambda x: ", ".join(x)),
                        n=("holiday", "count"),
                    )
                    .reset_index()
                )
                self.logger.debug(f"Aggregated {len(holidays)} weekly holiday entries")
                return holidays
            elif interval_type == "month":
                self.logger.debug("Processing monthly holiday aggregation")
                if not all(dt_transform["ds"].dt.day == 1):
                    self.logger.error(
                        "Monthly data does not start on first day of month"
                    )
                    raise ValueError(
                        "Monthly data should have first day of month as datestamp, e.g.'2020-01-01'"
                    )
                holidays["ds"] = holidays["ds"].dt.to_period("M").dt.to_timestamp()
                holidays = (
                    holidays.groupby(["ds", "country", "year"])["holiday"]
                    .agg(lambda x: ", ".join(x))
                    .reset_index()
                )
                self.logger.debug(f"Aggregated {len(holidays)} monthly holiday entries")
                return holidays
            else:
                self.logger.error(f"Invalid interval_type: {interval_type}")
                raise ValueError(
                    "Invalid interval_type. Must be 'day', 'week', or 'month'."
                )

        except Exception as e:
            self.logger.error(f"Error setting holidays: {str(e)}")
            raise

    def _apply_transformations(
        self, x: pd.Series, params: ChannelHyperparameters
    ) -> pd.Series:
        self.logger.debug("Applying transformations to series")
        x_adstock = self._apply_adstock(x, params)
        x_saturated = self._apply_saturation(x_adstock, params)
        return x_saturated

    def _apply_adstock(self, x: pd.Series, params: ChannelHyperparameters) -> pd.Series:
        self.logger.debug(
            f"Applying {self.hyperparameters.adstock} adstock transformation"
        )
        try:
            if self.hyperparameters.adstock == AdstockType.GEOMETRIC:
                return self._geometric_adstock(x, params.thetas[0])
            elif self.hyperparameters.adstock in [
                AdstockType.WEIBULL_CDF,
                AdstockType.WEIBULL_PDF,
            ]:
                return self._weibull_adstock(x, params.shapes[0], params.scales[0])
            else:
                self.logger.error(
                    f"Unsupported adstock type: {self.hyperparameters.adstock}"
                )
                raise ValueError(
                    f"Unsupported adstock type: {self.hyperparameters.adstock}"
                )
        except Exception as e:
            self.logger.error(f"Error applying adstock transformation: {str(e)}")
            raise

    @staticmethod
    def _geometric_adstock(x: pd.Series, theta: float) -> pd.Series:
        return x.ewm(alpha=1 - theta, adjust=False).mean()

    @staticmethod
    def _weibull_adstock(x: pd.Series, shape: float, scale: float) -> pd.Series:
        def weibull_pdf(t):
            return (
                (shape / scale)
                * ((t / scale) ** (shape - 1))
                * np.exp(-((t / scale) ** shape))
            )

        weights = [weibull_pdf(t) for t in range(1, len(x) + 1)]
        weights = weights / np.sum(weights)
        return np.convolve(x, weights[::-1], mode="full")[: len(x)]

    def _apply_saturation(
        self, x: pd.Series, params: ChannelHyperparameters
    ) -> pd.Series:
        self.logger.debug("Applying saturation transformation")
        try:
            alpha, gamma = params.alphas[0], params.gammas[0]
            return x**alpha / (x**alpha + gamma**alpha)
        except Exception as e:
            self.logger.error(f"Error applying saturation transformation: {str(e)}")
            raise
