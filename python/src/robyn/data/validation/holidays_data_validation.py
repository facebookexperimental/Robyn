#pyre-strict

from typing import List
from robyn.data.entities.holidays_data import HolidaysData
from robyn.data.entities.enums import ProphetVariableType
from robyn.data.validation.validation import Validation, ValidationResult
import numpy as np


class HolidaysDataValidation(Validation):
    def __init__(self, holidays_data: HolidaysData) -> None:
        self.holidays_data: HolidaysData = holidays_data

    def check_holidays(self) -> ValidationResult:
        """
        Check if the holidays data is valid.
        Check for missing (NA) values in the dt_holidays DataFrame.
        Check for missing required columns.
        Check for invalid characters (spaces) in the column names.

        Returns:
        - bool: True if the holidays data is valid, False otherwise.
        """
        dt_holidays = self.holidays_data.dt_holidays

        if dt_holidays is None:
            return ValidationResult(status=True, error_details={}, error_message="")

        error_details = {}
        error_message = ""

        # Check for NA values
        na_cols = dt_holidays.columns[dt_holidays.isnull().any()].tolist()
        if na_cols:
            na_details = []
            for col in na_cols:
                missing_count = dt_holidays[col].isnull().sum()
                missing_percentage = (missing_count / len(dt_holidays)) * 100
                na_details.append(f"{col} ({missing_count} | {missing_percentage:.2f}%)")
            
            error_details['missing'] = na_cols
            error_message += f"Dataset dt_holidays contains missing (NA) values. Missing values: {', '.join(na_details)}. "
 

        # For holidays data, we only need to check 'ds' and 'country' columns
        vars_to_check = ['ds', 'country']
        # Check for missing required columns
        missing_cols = [var for var in vars_to_check if var not in dt_holidays.columns]
        if missing_cols:
            error_details['missing_columns'] = missing_cols
            print(error_details)
            error_message += f"Missing required columns in holidays data: {', '.join(missing_cols)}. "

        # Check for invalid characters (spaces)
        invalid = [var for var in dt_holidays.columns if ' ' in var]
        if invalid:
            error_details['invalid'] = invalid
            error_message += f"Invalid column names (contains spaces) in holidays data: {', '.join(invalid)}. "

        return ValidationResult(
            status=not error_details,
            error_details=error_details,
            error_message=error_message
        )

    def check_prophet(self) -> ValidationResult:
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

        try:
            if dt_holidays is None or prophet_vars is None:
                return ValidationResult(status=True, error_details={}, error_message="")

            if  ProphetVariableType.HOLIDAY not in prophet_vars:
                if prophet_country is not None:
                    warning_message = f"Warning: Input 'prophet_country' is defined as {prophet_country} but 'holiday' is not setup within 'prophet_vars' parameter. "
                    # TODO: Print warning message
                prophet_country = None

            if ProphetVariableType.HOLIDAY in prophet_vars and (
                prophet_country is None or 
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
        except Exception as e:
            print(e)
            return ValidationResult(status=False, error_details=error_details, error_message=str(e))


    def validate(self) -> List[ValidationResult]:
        """
        Perform all validations and return the result.

        Returns:
        - ValidationResult: The result of the validation operation.
        """
        return [
            self.check_holidays(),
            self.check_prophet()
        ]