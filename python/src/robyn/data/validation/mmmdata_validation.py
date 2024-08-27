# pyre-strict
from typing import List

from robyn.data.entities.mmmdata import MMMData
from robyn.data.validation.validation import Validation, ValidationResult


class MMMDataValidation(Validation):
    def __init__(self, mmm_data: MMMData) -> None:
        self.mmm_data: MMMData = mmm_data

    def check_missing_and_infinite(self) -> ValidationResult:
        """
        Check for missing (NA) values and infinite (Inf) values in the DataFrame.
        
        :return: validation result with error_details containing a dictionary with keys 'missing' and 'infinite', each containing a list of column names with missing or infinite values.
        """
        pass
        #raise NotImplementedError("Not yet implemented")
    
    def check_no_variance(self) -> ValidationResult:
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
    
    def check_date_variable(self) -> ValidationResult:
        """
        Checks if the date variable is correct.
        
        :return: True if the date variable is valid, False otherwise.
        """
        raise NotImplementedError("Not yet implemented")
    
    def check_dependent_variables(self) -> ValidationResult:
        """
        Checks if dependent variables are valid.
        
        :return: True if the dependent variables are valid, False otherwise.
        """
        raise NotImplementedError("Not yet implemented")
    
    def validate(self) -> List[ValidationResult]:
        """
        Perform all validations and return the results.
        
        :return: A dictionary containing the results of all validations.
        """
        #raise NotImplementedError("Not yet implemented")
        pass
