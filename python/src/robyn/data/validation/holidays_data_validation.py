#pyre-strict

from typing import List
from robyn.data.entities.holidays_data import HolidaysData
from robyn.data.entities.enums import ProphetVariableType
from robyn.data.validation.validation import Validation, ValidationResult


class HolidaysDataValidation(Validation):
    def __init__(self, holidays_data: HolidaysData) -> None:
        self.holidays_data: HolidaysData = holidays_data

    def check_holidays(self) -> ValidationResult:
        """
        Check if the holidays data is valid.

        Returns:
        - bool: True if the holidays data is valid, False otherwise.
        """
        raise NotImplementedError("Not yet implemented")

    def check_prophet(self, holidays_data: HolidaysData) -> ValidationResult:
        """
        Check if the Prophet model is valid for the given data.

        Returns:
        - bool: True if the Prophet model is valid, False otherwise.
        """
        dt_holidays = self.holidays_data.dt_holidays
        prophet_vars = self.holidays_data.prophet_vars
        prophet_signs = self.holidays_data.prophet_signs
        prophet_country = self.holidays_data.prophet_country

        error_details = {}
        error_message = ""

        if dt_holidays is None or prophet_vars is None:
            return ValidationResult(status=True, error_details={}, error_message="")

        if  ProphetVariableType.HOLIDAY not in prophet_vars:
            if prophet_country is not None:
                warning_message += f"Warning: Input 'prophet_country' is defined as {prophet_country} but 'holiday' is not setup within 'prophet_vars' parameter. "
                # TODO: Print warning message
            prophet_country = None

        if ProphetVariableType.HOLIDAY in prophet_vars and (
            prophet_country is None or 
            len(prophet_country) > 1 or 
            prophet_country not in dt_holidays['country'].unique()
        ):
            available_countries = ", ".join(dt_holidays['country'].unique())
            error_details['prophet_country'] = f"Invalid prophet_country. Available countries: {available_countries}"
            error_message += f"Invalid prophet_country. "
            return ValidationResult(
                status=not bool(error_details),
                error_details=error_details,
                error_message=error_message
            )
        
        if prophet_signs is None:
            prophet_signs = ["default"] * len(prophet_vars)
        
        if len(prophet_signs) == 1:
            prophet_signs = prophet_signs * len(prophet_vars)

        if len(prophet_signs) != len(prophet_vars):
            error_details['prophet_signs_length'] = "prophet_signs must have same length as prophet_vars"
            error_message += "Mismatch in prophet_signs and prophet_vars length. "

        return ValidationResult(
            status=not bool(error_details),
            error_details=error_details,
            error_message=error_message.strip()
        )


    def validate(self) -> List[ValidationResult]:
        """
        Perform all validations and return the result.

        Returns:
        - ValidationResult: The result of the validation operation.
        """
        return [
            # self.check_holidays(),
            self.check_prophet(self.holidays_data)
        ]