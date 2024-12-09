# pyre-strict

import logging
from typing import List
from robyn.data.entities.holidays_data import HolidaysData
from robyn.data.entities.enums import ProphetVariableType
from robyn.data.validation.validation import Validation, ValidationResult
import numpy as np


class HolidaysDataValidation(Validation):
    def __init__(self, holidays_data: HolidaysData) -> None:
        self.holidays_data: HolidaysData = holidays_data
        self.logger = logging.getLogger(__name__)
        self.logger.debug(
            "Initializing HolidaysDataValidation with holidays_data: %s",
            holidays_data.__class__.__name__,
        )

    def check_holidays(self) -> ValidationResult:
        """
        Check if the holidays data is valid.
        Check for missing (NA) values in the dt_holidays DataFrame.
        Check for missing required columns.
        Check for invalid characters (spaces) in the column names.

        Returns:
        - ValidationResult: A ValidationResult object containing the status (True if the holidays data is valid, False otherwise),
            error details, and error message.
        """
        self.logger.debug("Starting holidays data validation check")
        dt_holidays = self.holidays_data.dt_holidays

        if dt_holidays is None:
            self.logger.info("No holidays data provided, skipping validation")
            return ValidationResult(status=True, error_details={}, error_message="")

        self.logger.debug(
            "Validating holidays DataFrame with shape: %s", dt_holidays.shape
        )
        error_details = {}
        error_message = ""

        # Check for NA values
        self.logger.debug("Checking for NA values in holidays data")
        na_cols = dt_holidays.columns[dt_holidays.isnull().any()].tolist()
        if na_cols:
            na_details = []
            for col in na_cols:
                missing_count = dt_holidays[col].isnull().sum()
                missing_percentage = (missing_count / len(dt_holidays)) * 100
                na_details.append(
                    f"{col} ({missing_count} | {missing_percentage:.2f}%)"
                )
                self.logger.warning(
                    "Column '%s' contains %d missing values (%.2f%%)",
                    col,
                    missing_count,
                    missing_percentage,
                )

            error_details["missing"] = na_cols
            error_message += f"Dataset dt_holidays contains missing (NA) values. Missing values: {', '.join(na_details)}. "

        # Check for missing required columns
        self.logger.debug("Checking for required columns presence")
        vars_to_check = ["ds", "country"]
        missing_cols = [var for var in vars_to_check if var not in dt_holidays.columns]
        if missing_cols:
            self.logger.error(
                "Required columns missing in holidays data: %s", missing_cols
            )
            error_details["missing_columns"] = missing_cols
            error_message += f"Missing required columns in holidays data: {', '.join(missing_cols)}. "

        # Check for invalid characters (spaces)
        self.logger.debug("Checking for invalid characters in column names")
        invalid = [var for var in dt_holidays.columns if " " in var]
        if invalid:
            self.logger.error(
                "Invalid column names found (contains spaces): %s", invalid
            )
            error_details["invalid"] = invalid
            error_message += f"Invalid column names (contains spaces) in holidays data: {', '.join(invalid)}. "

        validation_result = ValidationResult(
            status=not error_details,
            error_details=error_details,
            error_message=error_message,
        )
        self.logger.info(
            "Holidays validation completed. Status: %s", validation_result.status
        )
        return validation_result

    def check_prophet(self) -> ValidationResult:
        """
        Check if the Prophet model is valid for the given data.

        Returns:
        - ValidationResult: A ValidationResult object containing the status (True if the holidays data is valid, False otherwise),
            error details, and error message.
        """
        self.logger.debug("Starting Prophet model validation check")
        dt_holidays = self.holidays_data.dt_holidays
        prophet_vars = self.holidays_data.prophet_vars
        prophet_signs = self.holidays_data.prophet_signs
        prophet_country = self.holidays_data.prophet_country

        self.logger.debug(
            "Prophet validation parameters - vars: %s, signs: %s, country: %s",
            prophet_vars,
            prophet_signs,
            prophet_country,
        )

        error_details = {}
        error_message = ""

        try:
            if dt_holidays is None or prophet_vars is None:
                self.logger.info(
                    "No holidays data or prophet variables provided, skipping validation"
                )
                return ValidationResult(status=True, error_details={}, error_message="")

            if ProphetVariableType.HOLIDAY not in prophet_vars:
                if prophet_country is not None:
                    warning_message = f"Warning: Input 'prophet_country' is defined as {prophet_country} but 'holiday' is not setup within 'prophet_vars' parameter."
                    self.logger.warning(warning_message)
                prophet_country = None

            if ProphetVariableType.HOLIDAY in prophet_vars:
                self.logger.debug(
                    "Validating prophet_country against available countries"
                )
                if prophet_country is None:
                    self.logger.error(
                        "prophet_country is None but HOLIDAY type is specified in prophet_vars"
                    )
                elif prophet_country not in dt_holidays["country"].unique():
                    available_countries = dt_holidays["country"].unique()
                    self.logger.error(
                        "Invalid prophet_country '%s'. Available countries: %s",
                        prophet_country,
                        available_countries,
                    )
                    error_details["prophet_country"] = (
                        f"Invalid prophet_country. Available countries: {', '.join(available_countries)}"
                    )
                    error_message += f"Invalid prophet_country. "
                    return ValidationResult(
                        status=False,
                        error_details=error_details,
                        error_message=error_message,
                    )

            if prophet_signs is None:
                self.logger.debug("Initializing default prophet_signs")
                prophet_signs = ["default"] * len(prophet_vars)

            if len(prophet_signs) == 1:
                self.logger.debug(
                    "Expanding single prophet_sign to match prophet_vars length"
                )
                prophet_signs = prophet_signs * len(prophet_vars)

            if len(prophet_signs) != len(prophet_vars):
                self.logger.error(
                    "Length mismatch: prophet_signs (%d) != prophet_vars (%d)",
                    len(prophet_signs),
                    len(prophet_vars),
                )
                error_details["prophet_signs_length"] = (
                    "prophet_signs must have same length as prophet_vars"
                )
                error_message += "Mismatch in prophet_signs and prophet_vars length. "

            validation_result = ValidationResult(
                status=not bool(error_details),
                error_details=error_details,
                error_message=error_message.strip(),
            )
            self.logger.info(
                "Prophet validation completed. Status: %s", validation_result.status
            )
            return validation_result

        except Exception as e:
            self.logger.error(
                "Prophet validation failed with error: %s", str(e), exc_info=True
            )
            return ValidationResult(
                status=False, error_details=error_details, error_message=str(e)
            )

    def validate(self) -> List[ValidationResult]:
        """
        Perform all validations and return the result.

        Returns:
        - ValidationResult: The result of the validation operation.
        """
        self.logger.info("Starting complete validation process")
        results = [self.check_holidays(), self.check_prophet()]
        self.logger.info(
            "Validation complete. Overall status: %s",
            all(result.status for result in results),
        )
        return results
