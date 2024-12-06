import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.robyn.data.entities.holidays_data import HolidaysData
from src.robyn.data.validation.holidays_data_validation import HolidaysDataValidation
from src.robyn.data.entities.enums import ProphetVariableType, ProphetSigns


@pytest.fixture
def sample_holidays_data():
    dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
    df = pd.DataFrame({"ds": dates, "country": "US", "holiday": "New Year"})
    return HolidaysData(
        dt_holidays=df,
        prophet_country="US",
        prophet_vars=[ProphetVariableType.HOLIDAY],
        prophet_signs=[ProphetSigns.POSITIVE],
    )


def test_check_holidays_no_issues(sample_holidays_data):
    validator = HolidaysDataValidation(sample_holidays_data)
    result = validator.check_holidays()
    assert result.status == True
    assert not result.error_details
    assert not result.error_message


def test_check_holidays_with_missing_values(sample_holidays_data):
    sample_holidays_data.dt_holidays.loc[0, "holiday"] = np.nan
    validator = HolidaysDataValidation(sample_holidays_data)
    result = validator.check_holidays()
    assert result.status == False
    assert "missing" in result.error_details
    assert "holiday" in result.error_details["missing"]


def test_check_holidays_missing_required_column(sample_holidays_data):
    sample_holidays_data.dt_holidays = sample_holidays_data.dt_holidays.drop(
        "country", axis=1
    )
    validator = HolidaysDataValidation(sample_holidays_data)
    result = validator.check_holidays()
    assert result.status == False
    assert "missing_columns" in result.error_details
    assert "country" in result.error_details["missing_columns"]


def test_check_holidays_invalid_column_name(sample_holidays_data):
    sample_holidays_data.dt_holidays = sample_holidays_data.dt_holidays.rename(
        columns={"country": "country name"}
    )
    validator = HolidaysDataValidation(sample_holidays_data)
    result = validator.check_holidays()
    assert result.status == False
    assert "invalid" in result.error_details
    assert "country name" in result.error_details["invalid"]


def test_check_prophet_valid_input(sample_holidays_data):
    validator = HolidaysDataValidation(sample_holidays_data)
    result = validator.check_prophet()
    assert result.status == True
    assert not result.error_details
    assert not result.error_message


def test_validate_all_checks_including_prophet(sample_holidays_data):
    validator = HolidaysDataValidation(sample_holidays_data)
    results = validator.validate()
    assert all(result.status for result in results)


def test_validate_with_issues(sample_holidays_data):
    sample_holidays_data.dt_holidays.loc[0, "holiday"] = np.nan
    sample_holidays_data.dt_holidays = sample_holidays_data.dt_holidays.rename(
        columns={"country": "country_name"}
    )
    validator = HolidaysDataValidation(sample_holidays_data)
    results = validator.validate()
    assert not all(result.status for result in results)
    assert any("missing" in result.error_details for result in results)
