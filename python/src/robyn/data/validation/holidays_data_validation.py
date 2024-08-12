#pyre-strict

from typing import List
from robyn.data.entities.holidays_data import HolidaysData
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
        raise NotImplementedError("Not yet implemented")

    def validate(self) -> List[ValidationResult]:
        """
        Perform all validations and return the result.

        Returns:
        - ValidationResult: The result of the validation operation.
        """
        raise NotImplementedError("Not yet implemented")