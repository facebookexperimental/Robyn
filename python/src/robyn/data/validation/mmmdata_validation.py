# pyre-strict

from typing import List, Dict, Any, Optional
from collections import Counter
from data.entities.mmmdata_collection import MMMDataCollection, TimeWindow
import pandas as pd
import numpy as np
import re

from robyn.data.entities.mmmdata import MMMData


class MMMDataValidation:
    def __init__(self, mmm_data: MMMData) -> None:
        self.mmm_data: MMMData = mmm_data

    def check_missing_and_infinite(self) -> Dict[str, List[str]]:
        """
        Check for missing (NA) values and infinite (Inf) values in the DataFrame.

        :return: A dictionary with keys 'missing' and 'infinite', each containing a list of column names with issues.
        """
        missing_cols: List[str] = self.mmm_data.data.columns[
            self.mmm_data.data.isna().any()
        ].tolist()
        infinite_cols: List[str] = self.mmm_data.data.columns[
            np.isinf(self.mmm_data.data).any()
        ].tolist()
        return {"missing": missing_cols, "infinite": infinite_cols}

    def check_no_variance(self, mmmdata_collection: MMMDataCollection) -> List[str]:
        """
        Check for columns with no variance in the input dataframe.

        :return: A list of column names with no variance.
        """
        return self.mmm_data.data.columns[self.mmm_data.data.nunique() == 1].tolist()

    def check_variable_names(self) -> Dict[str, List[str]]:
        """
        Check variable names for duplicates and invalid characters.

        :return: A dictionary with keys 'duplicates' and 'invalid', each containing a list of problematic column names.
        """
        duplicates: List[str] = self.mmm_data.data.columns[
            self.mmm_data.data.columns.duplicated()
        ].tolist()
        invalid: List[str] = [
            col
            for col in self.mmm_data.data.columns
            if not re.match(r"^[a-zA-Z0-9_]+$", col)
        ]
        return {"duplicates": duplicates, "invalid": invalid}

    def check_datevar(self) -> bool:
        """
        Checks if the date variable is correct and returns a dictionary with the date variable name,
        interval type, and a tibble object of the input data.
        Parameters:
        - dt_input: The input data as a pandas DataFrame.
        - date_var: The name of the date variable to be checked. If set to "auto", the function will automatically detect the date variable.

        Returns:
        - output: A dictionary containing the following keys:
            - "date_var": The name of the date variable.
            - "dayInterval": The interval between the first two dates.
            - "intervalType": The type of interval (day, week, or month).
            - "dt_input": The input data as a pandas DataFrame.
        """
        date_var: str = self.mmm_data.mmmdata_spec.date_var
        return True

    def check_dependent_variables(self) -> bool:
        """
        Checks if dependent variables are valid.

        :return: True if the dependent variables are valid, False otherwise.
        """
        dep_var: Optional[str] = self.mmm_data.mmmdata_spec.dep_var
        if dep_var is None or dep_var not in self.mmm_data.data.columns:
            return False
        return self.mmm_data.data[dep_var].dtype in ["int64", "float64"]

    def check_context_variables(self) -> List[str]:
        """
        Checks if context variables are valid.

        :return: A list of invalid context variables.
        """
        if self.mmm_data.mmmdata_spec.context_vars is None:
            return []
        if self.mmm_data.mmmdata_spec.context_signs is None:
            context_signs = list()
        invalid_context_vars: List[str] = []
        return invalid_context_vars

    def check_paidmedia(self) -> List[str]:
        """
        Checks if paid media variables are valid.

        :return: A dictionary containing the following keys:
            - "paid_media_vars": A list of valid paid media variables.
            - "paid_media_signs": A list of signs for the valid paid media variables.
            - "mediaVarCount": The number of valid paid media variables.
        """
        paid_media_vars: Optional[List[str]] = (
            self.mmm_data.mmmdata_spec.paid_media_vars
        )
        paid_media_signs: Optional[List[str]] = (
            self.mmm_data.mmmdata_spec.paid_media_signs
        )
        invalid_variables: List[str] = []
        return invalid_variables

    def check_organic_variables(self) -> bool:
        """
        Checks if organic variables are valid.

        :return: A dictionary containing the following keys:
            - "organic_signs": A list of signs for the valid organic variables.
        """
        organic_vars: Optional[List[str]] = self.mmm_data.mmmdata_spec.organic_vars
        organic_signs: Optional[List[str]] = self.mmm_data.mmmdata_spec.organic_signs

        return True

    def check_factor_variables(self) -> bool:
        """
        Checks if factor variables are valid.

        :return: A dictionary containing the following keys:
            - "factor_vars": A list of valid factor variables.
            - "factor_signs": A list of signs for the valid factor variables.
        """
        factor_vars: Optional[List[str]] = self.mmm_data.mmmdata_spec.factor_vars
        organic_vars: Optional[List[str]] = self.mmm_data.mmmdata_spec.organic_vars
        context_vars: Optional[List[str]] = self.mmm_data.mmmdata_spec.context_vars

        if factor_vars is None:
            factor_vars = []

        return True

    def check_data_dimension(self, all_vars: List[str], rel: int = 10) -> bool:
        """
        Checks if the data has the correct dimension.

        :return: True if the data has the correct dimension, False otherwise.
        """
        return (
            self.mmm_data.data.shape[0]
            < (len(all_vars) * rel) & self.mmm_data.data.shape[1]
            <= 2
        )

    def check_model_training_window(
        self, all_media_vars: List[str], timeWindow: TimeWindow
    ) -> Dict[str, List[str]]:
        """
        Checks if the time windows are valid for training the model.

        :return: A dictionary containing the following keys:
            - "dt_input": The input data as a pandas DataFrame.
            - "window_start": The start date of the modeling window.
            - "rollingWindowStartWhich": The start date of the rolling window.
            - "refreshAddedStart": The start date of the refresh window.
            - "window_end": The end date of the modeling window.
            - "rollingWindowEndWhich": The end date of the rolling window.
            - "rollingWindowLength": The length of the rolling window.
        """
        window_start: Optional[int] = TimeWindow.window_start
        window_end: Optional[int] = TimeWindow.window_end
        invalid_variables = []
        output = {"invalid_variables": invalid_variables, "errors": []}
        return output

    def validate(self) -> Dict[str, Any]:
        """
        Perform all validations and return the results.

        :return: A dictionary containing the results of all validations.
        """
        return {
            "missing_and_infinite": self.check_missing_and_infinite(),
            "no_variance": self.check_no_variance(),
            "variable_names": self.check_variable_names(),
            "date_variable": self.check_date_variable(),
            "dependent_variables": self.check_dependent_variables(),
            "context_variables": self.check_context_variables(),
            "paidmedia": self.check_paidmedia(),
            "organic_variables": self.check_organic_variables(),
            "factor_variables": self.check_factor_variables(),
            "data_dimension": self.check_data_dimension(all_vars=[], rel=10),
            "model_training_window": self.check_model_training_window(
                all_media_vars=[], timeWindow=TimeWindow()
            ),
        }
