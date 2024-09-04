from datetime import datetime
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


class MMMDataCollection:
    def __init__(self):
        self.dt_input: pd.DataFrame = pd.DataFrame()
        self.dt_holidays: pd.DataFrame = pd.DataFrame()
        self.dt_mod: pd.DataFrame = pd.DataFrame()
        self.dt_modRollWind: pd.DataFrame = pd.DataFrame()
        self.xDecompAggPrev: Optional[pd.DataFrame] = None
        self.date_var: str = ""
        self.modNLS: Dict[str, Any] = {"results": None, "yhat": None, "plots": {}}
        self.dayInterval: int = 1
        self.intervalType: str = "day"
        self.dep_var: str = ""
        self.dep_var_type: str = ""
        self.prophet_vars: List[str] = []
        self.prophet_signs: List[str] = []
        self.prophet_country: Optional[str] = None
        self.context_vars: List[str] = []
        self.context_signs: List[str] = []
        self.paid_media_vars: List[str] = []
        self.paid_media_signs: List[str] = []
        self.paid_media_spends: List[str] = []
        self.paid_media_total: float = 0.0
        self.exposure_vars: List[str] = []
        self.organic_vars: List[str] = []
        self.organic_signs: List[str] = []
        self.all_media: List[str] = []
        self.all_ind_vars: List[str] = []
        self.factor_vars: List[str] = []
        self.unused_vars: List[str] = []
        self.window_start: datetime = datetime.now()
        self.rollingWindowStartWhich: int = 0
        self.window_end: datetime = datetime.now()
        self.rollingWindowEndWhich: int = 0
        self.rollingWindowLength: int = 0
        self.totalObservations: int = 0
        self.refreshAddedStart: datetime = datetime.now()
        self.adstock: str = "geometric"
        self.hyperparameters: Dict[str, Any] = {}
        self.calibration_input: Optional[pd.DataFrame] = None
        self.custom_params: Dict[str, Any] = {}
        self.version: str = ""

    def set_input_data(self, dt_input: pd.DataFrame) -> None:
        """Set the input data for the MMM."""
        self.dt_input = dt_input
        self.totalObservations = len(dt_input)

    def set_holidays_data(self, dt_holidays: pd.DataFrame) -> None:
        """Set the holidays data for the MMM."""
        self.dt_holidays = dt_holidays

    def set_date_variable(self, date_var: str) -> None:
        """Set the date variable name and calculate interval information."""
        self.date_var = date_var
        self._calculate_interval_info()

    def set_dependent_variable(self, dep_var: str, dep_var_type: str) -> None:
        """Set the dependent variable name and type."""
        self.dep_var = dep_var
        self.dep_var_type = dep_var_type

    def set_prophet_variables(
        self,
        prophet_vars: List[str],
        prophet_signs: Optional[List[str]] = None,
        prophet_country: Optional[str] = None,
    ) -> None:
        """Set the Prophet-related variables."""
        self.prophet_vars = [var.lower() for var in prophet_vars]
        if prophet_signs is None:
            self.prophet_signs = ["default"] * len(prophet_vars)
        else:
            self.prophet_signs = prophet_signs
        self.prophet_country = prophet_country

    def set_context_variables(
        self, context_vars: List[str], context_signs: List[str]
    ) -> None:
        """Set the context variables."""
        self.context_vars = context_vars
        self.context_signs = context_signs

    def set_paid_media_variables(
        self,
        paid_media_vars: List[str],
        paid_media_signs: List[str],
        paid_media_spends: List[str],
    ) -> None:
        """Set the paid media variables."""
        self.paid_media_vars = paid_media_vars
        self.paid_media_signs = paid_media_signs
        self.paid_media_spends = paid_media_spends
        self._calculate_paid_media_total()
        self.exposure_vars = [
            var for var in paid_media_vars if var not in paid_media_spends
        ]

    def set_organic_variables(
        self, organic_vars: List[str], organic_signs: List[str]
    ) -> None:
        """Set the organic variables."""
        self.organic_vars = organic_vars
        self.organic_signs = organic_signs

    def set_factor_variables(self, factor_vars: List[str]) -> None:
        """Set the factor variables."""
        self.factor_vars = factor_vars

    def set_window(self, window_start: str, window_end: str) -> None:
        """Set the modeling window."""
        self.window_start = pd.to_datetime(window_start)
        self.window_end = pd.to_datetime(window_end)
        self._calculate_window_info()

    def set_adstock(self, adstock: str) -> None:
        """Set the adstock type."""
        self.adstock = adstock

    def set_hyperparameters(self, hyperparameters: Dict[str, Any]) -> None:
        """Set the hyperparameters."""
        self.hyperparameters = hyperparameters

    def set_calibration_input(self, calibration_input: pd.DataFrame) -> None:
        """Set the calibration input data."""
        self.calibration_input = calibration_input

    def set_custom_params(self, custom_params: Dict[str, Any]) -> None:
        """Set custom parameters."""
        self.custom_params = custom_params

    def prepare_modeling_data(self) -> None:
        """Prepare the data for modeling."""
        print("Starting prepare_modeling_data")
        self._prepare_dt_mod()
        print(f"After _prepare_dt_mod, dt_mod shape: {self.dt_mod.shape}")
        self._calculate_window_info()
        print(
            f"After _calculate_window_info, window_start: {self.window_start}, window_end: {self.window_end}"
        )
        self._prepare_dt_modRollWind()
        print(
            f"After _prepare_dt_modRollWind, dt_modRollWind shape: {self.dt_modRollWind.shape}"
        )
        self._calculate_all_variables()
        self._fit_spend_exposure()
        print("Finished prepare_modeling_data")

    def _fit_spend_exposure(self) -> None:
        """Fit nonlinear models for media spend and exposure and display plots in new windows."""
        print("Starting _fit_spend_exposure method")
        print(f"dt_modRollWind shape: {self.dt_modRollWind.shape}")
        print(f"dt_modRollWind columns: {self.dt_modRollWind.columns}")

        if self.dt_modRollWind.empty:
            print("Error: dt_modRollWind is empty")
            return

        exposure_selector = [
            var != spend
            for var, spend in zip(self.paid_media_vars, self.paid_media_spends)
        ]
        print(f"exposure_selector: {exposure_selector}")

        if any(exposure_selector):
            results = []
            yhat_collect = []

            for i, (var, spend) in enumerate(
                zip(self.paid_media_vars, self.paid_media_spends)
            ):
                if exposure_selector[i]:
                    print(f"Processing {var} (spend: {spend})")

                    if spend not in self.dt_modRollWind.columns:
                        print(f"Error: {spend} not found in dt_modRollWind")
                        continue
                    if var not in self.dt_modRollWind.columns:
                        print(f"Error: {var} not found in dt_modRollWind")
                        continue

                    spend_data = self.dt_modRollWind[spend]
                    exposure_data = self.dt_modRollWind[var]

                    if spend_data.empty or exposure_data.empty:
                        print(f"Error: spend_data or exposure_data is empty for {var}")
                        continue

                    print(f"spend_data range: {spend_data.min()} - {spend_data.max()}")
                    print(
                        f"exposure_data range: {exposure_data.min()} - {exposure_data.max()}"
                    )

                    try:
                        # Fit Michaelis-Menten model
                        popt_nls, _ = curve_fit(
                            self._michaelis_menten,
                            spend_data,
                            exposure_data,
                            p0=[max(exposure_data), np.median(spend_data)],
                            bounds=([0, 0], [np.inf, np.inf]),
                        )

                        # Fit linear model
                        popt_lm, _ = curve_fit(
                            self._linear_model, spend_data, exposure_data
                        )

                        # Calculate R-squared for both models
                        rsq_nls = self._calculate_rsq(
                            exposure_data, self._michaelis_menten(spend_data, *popt_nls)
                        )
                        rsq_lm = self._calculate_rsq(
                            exposure_data, self._linear_model(spend_data, *popt_lm)
                        )

                        results.append(
                            {
                                "channel": var,
                                "Vmax": popt_nls[0],
                                "Km": popt_nls[1],
                                "rsq_nls": rsq_nls,
                                "rsq_lm": rsq_lm,
                                "coef_lm": popt_lm[0],
                            }
                        )

                        # Create plot in a new window
                        plt.figure(figsize=(10, 6))
                        plt.scatter(spend_data, exposure_data, label="Data")
                        spend_range = np.linspace(min(spend_data), max(spend_data), 100)
                        plt.plot(
                            spend_range,
                            self._michaelis_menten(spend_range, *popt_nls),
                            label="NLS fit",
                        )
                        plt.plot(
                            spend_range,
                            self._linear_model(spend_range, *popt_lm),
                            label="Linear fit",
                        )
                        plt.xlabel(f"Spend [{spend}]")
                        plt.ylabel(f"Exposure [{var}]")
                        plt.title(f"Spend vs Exposure for {var}")
                        plt.legend()
                        plt.show(
                            block=False
                        )  # Show the plot without blocking execution

                        # Collect yhat data
                        yhat_collect.append(
                            pd.DataFrame(
                                {
                                    "channel": var,
                                    "yhatNLS": self._michaelis_menten(
                                        spend_data, *popt_nls
                                    ),
                                    "yhatLM": self._linear_model(spend_data, *popt_lm),
                                    "y": exposure_data,
                                    "x": spend_data,
                                }
                            )
                        )
                    except Exception as e:
                        print(f"Error fitting models for {var}: {str(e)}")

            if results:
                self.modNLS["results"] = pd.DataFrame(results)
            if yhat_collect:
                self.modNLS["yhat"] = pd.concat(yhat_collect, ignore_index=True)

        print("Finished _fit_spend_exposure method")
        print(f"modNLS contents: {self.modNLS}")
        print("Plots have been displayed in new windows")

        # Keep the plot windows open
        plt.show()

    @staticmethod
    def _michaelis_menten(x, Vmax, Km):
        """Michaelis-Menten function."""
        return Vmax * x / (Km + x)

    @staticmethod
    def _linear_model(x, a):
        """Simple linear model."""
        return a * x

    @staticmethod
    def _calculate_rsq(y_true, y_pred):
        """Calculate R-squared."""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot)

    def _calculate_interval_info(self) -> None:
        """Calculate the interval information based on the date variable."""
        if self.date_var and not self.dt_input.empty:
            dates = pd.to_datetime(self.dt_input[self.date_var])
            self.dayInterval = (dates.iloc[1] - dates.iloc[0]).days
            if self.dayInterval == 1:
                self.intervalType = "day"
            elif self.dayInterval == 7:
                self.intervalType = "week"
            elif 28 <= self.dayInterval <= 31:
                self.intervalType = "month"
            else:
                raise ValueError(f"Unsupported interval: {self.dayInterval} days")

    def _calculate_paid_media_total(self) -> None:
        """Calculate the total paid media spend."""
        if not self.dt_input.empty and self.paid_media_spends:
            self.paid_media_total = self.dt_input[self.paid_media_spends].sum().sum()

    def _calculate_window_info(self) -> None:
        """Calculate window-related information."""
        if not self.dt_mod.empty and self.date_var:
            self.rollingWindowStartWhich = self.dt_mod[
                self.dt_mod["ds"] >= self.window_start
            ].index[0]
            self.rollingWindowEndWhich = self.dt_mod[
                self.dt_mod["ds"] <= self.window_end
            ].index[-1]
            self.rollingWindowLength = (
                self.rollingWindowEndWhich - self.rollingWindowStartWhich + 1
            )

    def _prepare_dt_mod(self) -> None:
        """Prepare the dt_mod DataFrame."""
        if not self.dt_input.empty:
            self.dt_mod = self.dt_input.copy()
            self.dt_mod.rename(
                columns={self.date_var: "ds", self.dep_var: "dep_var"}, inplace=True
            )
            self.dt_mod["ds"] = pd.to_datetime(self.dt_mod["ds"])
            self.dt_mod = self.dt_mod.sort_values("ds")

    def _prepare_dt_modRollWind(self) -> None:
        """Prepare the dt_modRollWind DataFrame."""
        if not self.dt_mod.empty:
            self.dt_modRollWind = self.dt_mod.iloc[
                self.rollingWindowStartWhich : self.rollingWindowEndWhich + 1
            ].copy()

    def _calculate_all_variables(self) -> None:
        """Calculate all_media and all_ind_vars."""
        self.all_media = self.paid_media_spends + self.organic_vars
        self.all_ind_vars = self.prophet_vars + self.context_vars + self.all_media
        self.unused_vars = [
            col
            for col in self.dt_input.columns
            if col not in [self.date_var, self.dep_var] + self.all_ind_vars
        ]

    def get_model_data(self) -> pd.DataFrame:
        """Get the prepared modeling data."""
        return self.dt_modRollWind

    def get_full_data(self) -> pd.DataFrame:
        """Get the full input data."""
        return self.dt_input

    def get_holidays_data(self) -> pd.DataFrame:
        """Get the holidays data."""
        return self.dt_holidays

    def set_version(self, version: str) -> None:
        """Set the version information."""
        self.version = version

    def __str__(self) -> str:
        """Return a string representation of the MMMDataCollection."""
        return f"""
Total Observations: {self.totalObservations} ({self.intervalType}s)
Input Table Columns ({len(self.dt_input.columns)}):
  Date: {self.date_var}
  Dependent: {self.dep_var} [{self.dep_var_type}]
  Paid Media: {', '.join(self.paid_media_vars)}
  Paid Media Spend: {', '.join(self.paid_media_spends)}
  Context: {', '.join(self.context_vars)}
  Organic: {', '.join(self.organic_vars)}
  Prophet (Auto-generated): {self._get_prophet_info()}
  Unused variables: {', '.join(self.unused_vars) if self.unused_vars else 'None'}

Date Range: {self.dt_input[self.date_var].min()} : {self.dt_input[self.date_var].max()}
Model Window: {self.window_start} : {self.window_end} ({self.rollingWindowLength} {self.intervalType}s)
With Calibration: {self.calibration_input is not None}
Custom parameters: {self._get_custom_params_info()}

Adstock: {self.adstock}
{self._get_hyperparameters_info()}
"""

    def _get_prophet_info(self) -> str:
        """Get Prophet information for string representation."""
        if self.prophet_vars:
            return f"{', '.join(self.prophet_vars)} on {self.prophet_country if self.prophet_country else 'data'}"
        return "Deactivated"

    def _get_custom_params_info(self) -> str:
        """Get custom parameters information for string representation."""
        if self.custom_params:
            return "\n" + "\n".join(
                [f"  {k}: {v}" for k, v in self.custom_params.items()]
            )
        return "None"

    def _get_hyperparameters_info(self) -> str:
        """Get hyperparameters information for string representation."""
        if self.hyperparameters:
            return f"Hyper-parameters ranges:\n" + "\n".join(
                [f"  {k}: {v}" for k, v in self.hyperparameters.items()]
            )
        return "Hyper-parameters: Not set yet"
