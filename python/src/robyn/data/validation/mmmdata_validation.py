# pyre-strict
from typing import List

from robyn.data.entities.mmmdata import MMMData
from robyn.data.validation.validation import Validation, ValidationResult
import pandas as pd
import numpy as np


class MMMDataValidation(Validation):
    def __init__(self, mmm_data: MMMData) -> None:
        self.mmm_data: MMMData = mmm_data

    def check_missing_and_infinite(self) -> ValidationResult:
        """
        Check for missing (NA) values and infinite (Inf) values in the DataFrame.
        
        :return: validation result with error_details containing a dictionary with keys 'missing' and 'infinite', each containing a list of column names with missing or infinite values.
        """
        dt_input = self.mmm_data.data
        
        missing_cols = dt_input.columns[dt_input.isnull().any()].tolist()
        infinite_cols = dt_input.columns[np.isinf(dt_input).any()].tolist()
        
        error_details = {}
        error_message = ""
        
        if missing_cols:
            error_details['missing'] = missing_cols
            error_message += f"Dataset contains missing (NA) values in columns: {', '.join(missing_cols)}. "
        
        if infinite_cols:
            error_details['infinite'] = infinite_cols
            error_message += f"Dataset contains infinite (Inf) values in columns: {', '.join(infinite_cols)}. "
        
        if error_message:
            error_message += "These values must be removed or fixed for Robyn to properly work."
        
        return ValidationResult(
            status=not bool(error_details),
            error_details=error_details,
            error_message=error_message
        )
    
    
    def check_no_variance(self) -> ValidationResult:
        """
        Check for columns with no variance in the input dataframe.
        
        :return: A list of column names with no variance.
        """
        dt_input = self.mmm_data.data
        
        no_variance_cols = dt_input.columns[dt_input.nunique() == 1].tolist()
        
        error_details = {}
        error_message = ""
        
        if no_variance_cols:
            error_details['no_variance'] = no_variance_cols
            error_message = f"There are {len(no_variance_cols)} column(s) with no-variance: {', '.join(no_variance_cols)}. Please remove the variable(s) to proceed."
        
        return ValidationResult(
            status=not bool(error_details),
            error_details=error_details,
            error_message=error_message
        )
    

    def check_variable_names(self) -> ValidationResult:
        """
        Check variable names for duplicates and invalid characters.
        
        :return: A dictionary with keys 'duplicates' and 'invalid', each containing a list of problematic column names.
        """

        mmmdata_spec = self.mmm_data.mmmdata_spec

        # Collect all variable names to check
        vars_to_check = [
            mmmdata_spec.dep_var,
            mmmdata_spec.date_var,
            mmmdata_spec.context_vars,
            mmmdata_spec.paid_media_spends,
            mmmdata_spec.organic_vars
        ]
        vars_to_check = [var for var in vars_to_check if var is not None]
        vars_to_check = [item for sublist in vars_to_check for item in (sublist if isinstance(sublist, list) else [sublist])]

        # Check for duplicates
        duplicates = [var for var in set(vars_to_check) if vars_to_check.count(var) > 1]

        # Check for invalid characters (spaces)
        invalid = [var for var in vars_to_check if ' ' in var]

        #prepare error messages
        error_details = {}
        error_message = ""
        if duplicates:
            error_details['duplicates'] = duplicates
            error_message += "Duplicate variable names present. "
        if invalid:
            error_details['invalid'] = invalid
            error_message += "Invalid variable names present. "


        return ValidationResult(
            # status is true if error_details is empty
            status=not error_details,
            error_details=error_details,
            error_message=error_message
        )
    
    
    def check_date_variable(self) -> ValidationResult:
        """
        Checks if the date variable is correct.
        
        :return: True if the date variable is valid, False otherwise.
        """
        mmmdata_spec = self.mmm_data.mmmdata_spec
        dt_input = self.mmm_data.data
        date_var = mmmdata_spec.date_var
        
        error_details = {}
        error_message = ""
        
        if date_var not in dt_input.columns:
            error_message = f"Date variable '{date_var}' not found in the input data."
        else:
            try:
                dt_input[date_var] = pd.to_datetime(dt_input[date_var])
                if not dt_input[date_var].is_monotonic_increasing:
                    error_message = f"Date variable '{date_var}' is not in ascending order."
            except ValueError:
                error_message = f"Date variable '{date_var}' contains invalid date values."
        
        if error_message:
            error_details['date_variable'] = error_message
        
        return ValidationResult(
            status=not bool(error_details),
            error_details=error_details,
            error_message=error_message
        )
    
    
    def check_dependent_variables(self) -> ValidationResult:
        """
        Checks if dependent variables are valid.
        
        :return: True if the dependent variables are valid, False otherwise.
        """
        mmmdata_spec = self.mmm_data.mmmdata_spec
        dt_input = self.mmm_data.data
        dep_var = mmmdata_spec.dep_var
        dep_var_type = mmmdata_spec.dep_var_type
        
        error_details = {}
        error_message = ""
        
        if dep_var not in dt_input.columns:
            error_message = f"Dependent variable '{dep_var}' not found in the input data."
        else:
            if not pd.api.types.is_numeric_dtype(dt_input[dep_var]):
                error_message = f"Dependent variable '{dep_var}' must be numeric."
            elif dep_var_type not in ["revenue", "conversion"]:
                error_message = f"Invalid dep_var_type '{dep_var_type}'. Must be 'revenue' or 'conversion'."
        
        if error_message:
            error_details['dependent_variable'] = error_message
        
        return ValidationResult(
            status=not bool(error_details),
            error_details=error_details,
            error_message=error_message
        )


    def validate(self) -> List[ValidationResult]:
        """
        Perform all validations and return the results.
        
        :return: A dictionary containing the results of all validations.
        """
        return [
            self.check_missing_and_infinite(),
            self.check_no_variance(),
            self.check_variable_names(),
            self.check_date_variable(),
            self.check_dependent_variables(),
        ]
