# pyre-strict

from typing import List, Dict, Any, Optional
from collections import Counter
from data.entities.mmmdata_collection import MMMDataCollection, TimeWindow
import pandas as pd
import numpy as np
from robyn.data.entities.mmmdata import MMMData
from robyn.data.validation.validation import Validation, ValidationResult


class MMMDataValidation(Validation):
    def __init__(self, mmm_data: MMMData) -> None:
        self.mmm_data: MMMData = mmm_data

    def check_missing_and_infinite(self) -> ValidationResult:
        """
        Check for missing (NA) values and infinite (Inf) values in the DataFrame.

        :return: A dictionary with keys 'missing' and 'infinite', each containing a list of column names with issues.
        """
        raise NotImplementedError("Not yet implemented")

    def check_no_variance(
        self, mmmdata_collection: MMMDataCollection
    ) -> ValidationResult:
        """
        Check for columns with no variance in the input dataframe.

        :return: A list of column names with no variance.
        """
        raise NotImplementedError("Not yet implemented")

    def check_variable_names(self) -> ValidationResult:
        """
        Check variable names for duplicates and invalid characters.

        :return: A dictionary with keys 'duplicates' and 'invalid', each containing a list of problematic column names.
        """
        raise NotImplementedError("Not yet implemented")

    def check_datevar(self) -> ValidationResult:
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
        raise NotImplementedError("Not yet implemented")

    def check_dependent_variables(self) -> ValidationResult:
        """
        Checks if dependent variables are valid.

        :return: True if the dependent variables are valid, False otherwise.
        """
        raise NotImplementedError("Not yet implemented")

    def check_context_variables(self) -> ValidationResult:
        """
        Checks if context variables are valid.

        :return: A list of invalid context variables.
        """
        raise NotImplementedError("Not yet implemented")

    def check_paidmedia(self) -> ValidationResult:
        """
        Checks if paid media variables are valid.

        :return: A dictionary containing the following keys:
            - "paid_media_vars": A list of valid paid media variables.
            - "paid_media_signs": A list of signs for the valid paid media variables.
            - "mediaVarCount": The number of valid paid media variables.
        """
        raise NotImplementedError("Not yet implemented")

    def check_organic_variables(self) -> ValidationResult:
        """
        Checks if organic variables are valid.

        :return: A dictionary containing the following keys:
            - "organic_signs": A list of signs for the valid organic variables.
        """
        organic_vars: Optional[List[str]] = self.mmm_data.mmmdata_spec.organic_vars
        organic_signs: Optional[List[str]] = self.mmm_data.mmmdata_spec.organic_signs
        raise NotImplementedError("Not yet implemented")

    def check_factor_variables(self) -> ValidationResult:
        """
        Checks if factor variables are valid.

        :return: A dictionary containing the following keys:
            - "factor_vars": A list of valid factor variables.
            - "factor_signs": A list of signs for the valid factor variables.
        """
        factor_vars: Optional[List[str]] = self.mmm_data.mmmdata_spec.factor_vars
        organic_vars: Optional[List[str]] = self.mmm_data.mmmdata_spec.organic_vars
        context_vars: Optional[List[str]] = self.mmm_data.mmmdata_spec.context_vars
        raise NotImplementedError("Not yet implemented")

    def check_data_dimension(
        self, all_vars: List[str], rel: int = 10
    ) -> ValidationResult:
        """
        Checks if the data has the correct dimension.

        :return: True if the data has the correct dimension, False otherwise.
        """
        raise NotImplementedError("Not yet implemented")

    def check_model_training_window(
        self, all_media_vars: List[str], timeWindow: TimeWindow
    ) -> ValidationResult:
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
        raise NotImplementedError("Not yet implemented")

    def validate(self) -> ValidationResult:
        """
        Perform all validations and return the results.

        :return: A dictionary containing the results of all validations.
        """
        raise NotImplementedError("Not yet implemented")
