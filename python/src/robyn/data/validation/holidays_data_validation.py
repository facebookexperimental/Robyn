from dataclasses import dataclass

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

    # Check if holidays_data is not None
    if holidays_data is None:
        is_valid = False
        errors.append("Holidays data cannot be None")

    # Check if prophet_vars is a valid list
    if holidays_data.prophet_vars is None or not isinstance(holidays_data.prophet_vars, list):
        is_valid = False
        errors.append("Prophet variables must be a list")

    # Check if prophet_signs is a valid list or None
    if holidays_data.prophet_signs is not None and not isinstance(holidays_data.prophet_signs, list):
        is_valid = False
        errors.append("Prophet signs must be a list or None")

    # Check if prophet_country is a valid string or None
    if holidays_data.prophet_country is not None and not isinstance(holidays_data.prophet_country, str):
        is_valid = False
        errors.append("Prophet country must be a string or None")

    # Check if day_interval is a valid integer or None
    if holidays_data.day_interval is not None and not isinstance(holidays_data.day_interval, int):
        is_valid = False
        errors.append("Day interval must be an integer or None")

    # Check if prophet_vars contains holiday and prophet_country is not None
    if "holiday" not in holidays_data.prophet_vars:
        if holidays_data.prophet_country is not None:
            warnings.append(f"Input 'prophet_country' is defined as {holidays_data.prophet_country} but 'holiday' is not setup within 'prophet_vars' parameter")
        holidays_data.prophet_country = None

    # Check if prophet_vars contains valid values
    opts = ["trend", "season", "monthly", "weekday", "holiday"]
    if not all(pv in opts for pv in holidays_data.prophet_vars):
        is_valid = False
        errors.append(f"Allowed values for `prophet_vars` are:  {opts}")

    # Check if prophet_vars contains weekday and day_interval > 7
    if "weekday" in holidays_data.prophet_vars and holidays_data.day_interval > 7:
        warnings.append("Ignoring prophet_vars = 'weekday' input given your data granularity")

    # Check if prophet_country is not in dt_holidays$country
    if "holiday" in holidays_data.prophet_vars and (holidays_data.prophet_country is None or holidays_data.prophet_country not in holidays_data.dt_holidays["country"].values):
        unique_countries = set(holidays_data.dt_holidays["country"].values)
        country_count = len(unique_countries)
        is_valid = False
        errors.append(f"You must provide 1 country code in 'prophet_country' input. {country_count} countries are included: {unique_countries} If your country is not available, manually include data to 'dt_holidays' or remove 'holidays' from 'prophet_vars' input.")

    # Check if prophet_signs is a valid list of strings
    if holidays_data.prophet_signs is None:
        prophet_signs = ["default"] * len(holidays_data.prophet_vars)
    else:
        prophet_signs = holidays_data.prophet_signs
    if not all(x in OPTS_PDN for x in prophet_signs):
        is_valid = False
        errors.append(f"Allowed values for 'prophet_signs' are: {', '.join(OPTS_PDN)}")

    # Check if prophet_signs has the same length as prophet_vars
    if len(prophet_signs) != len(holidays_data.prophet_vars):
        is_valid = False
        errors.append("'prophet_signs' must have the same length as 'prophet_vars'")

    return ValidationResult(is_valid, errors, warnings)
