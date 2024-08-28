# mmmdata_collection.py

from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


class MMMDataCollection:
    def __init__(self):
        self.dt_input: pd.DataFrame = pd.DataFrame()
        self.dt_holidays: pd.DataFrame = pd.DataFrame()
        self.dt_mod: pd.DataFrame = pd.DataFrame()
        self.dt_modRollWind: pd.DataFrame = pd.DataFrame()
        self.xDecompAggPrev: Optional[pd.DataFrame] = None
        self.date_var: str = ""
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
        prophet_signs: List[str],
        prophet_country: Optional[str],
    ) -> None:
        """Set the Prophet-related variables."""
        self.prophet_vars = prophet_vars
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

    def set_organic_variables(
        self, organic_vars: List[str], organic_signs: List[str]
    ) -> None:
        """Set the organic variables."""
        self.organic_vars = organic_vars
        self.organic_signs = organic_signs

    def set_factor_variables(self, factor_vars: List[str]) -> None:
        """Set the factor variables."""
        self.factor_vars = factor_vars

    def set_window(self, window_start: datetime, window_end: datetime) -> None:
        """Set the modeling window."""
        self.window_start = window_start
        self.window_end = window_end
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
        self._prepare_dt_mod()
        self._prepare_dt_modRollWind()
        self._calculate_all_variables()

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
        if not self.dt_input.empty and self.date_var:
            dates = pd.to_datetime(self.dt_input[self.date_var])
            self.rollingWindowStartWhich = (dates >= self.window_start).idxmax()
            self.rollingWindowEndWhich = (dates <= self.window_end).idxmax()
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
