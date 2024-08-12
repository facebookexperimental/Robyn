# pyre-strict

from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import re

from robyn.data.entities.mmmdata import MMMData
from robyn.data.validation.validation import Validation, ValidationResult


class MMMDataValidation(Validation):
    def __init__(self, mmm_data: MMMData) -> None:
        self.mmm_data: MMMData = mmm_data

    def check_missing_and_infinite(self) -> Dict[str, List[str]]:
        """
        Check for missing (NA) values and infinite (Inf) values in the DataFrame.
        
        :return: A dictionary with keys 'missing' and 'infinite', each containing a list of column names with issues.
        """
        missing_cols: List[str] = self.mmm_data.data.columns[self.mmm_data.data.isna().any()].tolist()
        infinite_cols: List[str] = self.mmm_data.data.columns[np.isinf(self.mmm_data.data).any()].tolist()
        return {"missing": missing_cols, "infinite": infinite_cols}

    def check_no_variance(self) -> List[str]:
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
        duplicates: List[str] = self.mmm_data.data.columns[self.mmm_data.data.columns.duplicated()].tolist()
        invalid: List[str] = [col for col in self.mmm_data.data.columns if not re.match(r'^[a-zA-Z0-9_]+$', col)]
        return {"duplicates": duplicates, "invalid": invalid}

    def check_date_variable(self) -> bool:
        """
        Checks if the date variable is correct.
        
        :return: True if the date variable is valid, False otherwise.
        """
        date_var: str = self.mmm_data.mmmdata_spec.date_var
        if date_var not in self.mmm_data.data.columns:
            return False
        try:
            pd.to_datetime(self.mmm_data.data[date_var])
            return True
        except ValueError:
            return False

    def check_dependent_variables(self) -> bool:
        """
        Checks if dependent variables are valid.
        
        :return: True if the dependent variables are valid, False otherwise.
        """
        dep_var: Optional[str] = self.mmm_data.mmmdata_spec.dep_var
        if dep_var is None or dep_var not in self.mmm_data.data.columns:
            return False
        return self.mmm_data.data[dep_var].dtype in ['int64', 'float64']

    def validate(self) -> ValidationResult:
        """
        Perform all validations and return the results.
        
        :return: A dictionary containing the results of all validations.
        """
        return {
            "missing_and_infinite": self.check_missing_and_infinite(),
            "no_variance": self.check_no_variance(),
            "variable_names": self.check_variable_names(),
            "date_variable": self.check_date_variable(),
            "dependent_variables": self.check_dependent_variables()
        }
