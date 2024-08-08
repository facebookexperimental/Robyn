from dataclasses import dataclass

from robyn.data.entities.holidays_data import HolidaysData

@dataclass
class ValidationResult:
    is_valid: bool
    errors: list[str]
    warnings: list[str]

def check_prophet(self, holidays_data: HolidaysData) -> ValidationResult:
    """
    Check if the Prophet model is valid for the given data.

    Parameters:
    - holidays_data (HolidaysData): The holidays data to check.

    Returns:
    - ValidationResult: The result of the validation.
    """

    is_valid = True
    errors = []
    warnings = []
    prophet_signs = None

    return ValidationResult(is_valid, errors, warnings)
