import pytest
import pandas as pd
from datetime import datetime, timedelta
from src.robyn.data.entities.calibration_input import CalibrationInput, ChannelCalibrationData
from src.robyn.data.entities.mmmdata import MMMData
from src.robyn.data.validation.calibration_input_validation import CalibrationInputValidation
from src.robyn.data.entities.enums import DependentVarType, CalibrationScope


@pytest.fixture
def sample_mmmdata():
    data = pd.DataFrame(
        {
            "date": pd.date_range(start="2022-01-01", periods=10),
            "revenue": [100, 120, 110, 130, 140, 150, 160, 170, 180, 190],
            "tv_spend": [50, 60, 55, 65, 70, 75, 80, 85, 90, 95],
            "radio_spend": [30, 35, 32, 38, 40, 42, 45, 48, 50, 52],
            "temperature": [20, 22, 21, 23, 24, 25, 26, 27, 28, 29],
        }
    )

    mmm_data_spec = MMMData.MMMDataSpec(
        dep_var="revenue", date_var="date", paid_media_spends=["tv_spend", "radio_spend"], context_vars=["temperature"]
    )

    return MMMData(data, mmm_data_spec)


# Add a fixture for multi-channel test cases
@pytest.fixture
def sample_multichannel_calibration_input():
    return CalibrationInput(
        channel_data={
            ("tv_spend", "radio_spend"): ChannelCalibrationData(
                lift_start_date=datetime(2022, 1, 1),
                lift_end_date=datetime(2022, 1, 5),
                lift_abs=3000,
                spend=520,  # Combined spend of both channels
                confidence=0.9,
                metric=DependentVarType.REVENUE,
                calibration_scope=CalibrationScope.IMMEDIATE,
            ),
            ("tv_spend",): ChannelCalibrationData(
                lift_start_date=datetime(2022, 1, 6),
                lift_end_date=datetime(2022, 1, 10),
                lift_abs=1000,
                spend=300,
                confidence=0.85,
                metric=DependentVarType.REVENUE,
                calibration_scope=CalibrationScope.IMMEDIATE,
            ),
        }
    )


@pytest.fixture
def sample_calibration_input():
    return CalibrationInput(
        channel_data={
            ("tv_spend"): ChannelCalibrationData(
                lift_start_date=datetime(2022, 1, 1),
                lift_end_date=datetime(2022, 1, 5),
                lift_abs=1000,
                spend=300,
                confidence=0.9,
                metric=DependentVarType.REVENUE,
                calibration_scope=CalibrationScope.IMMEDIATE,
            ),
            ("radio_spend"): ChannelCalibrationData(
                lift_start_date=datetime(2022, 1, 6),
                lift_end_date=datetime(2022, 1, 10),
                lift_abs=2000,
                spend=220,
                confidence=0.85,
                metric=DependentVarType.REVENUE,
                calibration_scope=CalibrationScope.IMMEDIATE,
            ),
        }
    )


# this function is to create the sample data for each unit test as we cannot directly modify the fixture data (as calibration dataclass is marked frozen)
def create_modified_calibration_input(original_input, channel_name, **kwargs):
    # Convert channel_name to tuple if it's a string
    channel_key = (channel_name,) if isinstance(channel_name, str) else channel_name

    # Get original data using tuple key
    original_channel_data = original_input.channel_data[channel_key]

    # Create new channel data with updates
    new_channel_data = ChannelCalibrationData(
        lift_start_date=kwargs.get("lift_start_date", original_channel_data.lift_start_date),
        lift_end_date=kwargs.get("lift_end_date", original_channel_data.lift_end_date),
        lift_abs=kwargs.get("lift_abs", original_channel_data.lift_abs),
        spend=kwargs.get("spend", original_channel_data.spend),
        confidence=kwargs.get("confidence", original_channel_data.confidence),
        metric=kwargs.get("metric", original_channel_data.metric),
        calibration_scope=kwargs.get("calibration_scope", original_channel_data.calibration_scope),
    )

    # Create new dictionary with updated data
    new_channel_data_dict = original_input.channel_data.copy()
    new_channel_data_dict[channel_key] = new_channel_data

    return CalibrationInput(channel_data=new_channel_data_dict)


def test_check_calibration_valid(sample_mmmdata, sample_calibration_input):
    validator = CalibrationInputValidation(
        sample_mmmdata, sample_calibration_input, window_start=datetime(2022, 1, 1), window_end=datetime(2022, 1, 10)
    )
    result = validator.check_calibration()
    assert result.status == True
    assert not result.error_details
    assert not result.error_message


def test_check_date_range_invalid(sample_mmmdata, sample_calibration_input):
    # Use tuple for channel key
    new_calibration_input = create_modified_calibration_input(
        sample_calibration_input, ("tv_spend",), lift_start_date=datetime(2021, 12, 31)  # Changed to tuple
    )

    validator = CalibrationInputValidation(
        sample_mmmdata, new_calibration_input, window_start=datetime(2022, 1, 1), window_end=datetime(2022, 1, 10)
    )
    result = validator._check_date_range()
    assert result.status == False
    # Check using tuple key in error_details
    assert ("tv_spend",) in result.error_details
    assert "outside the modeling window" in result.error_message


def test_check_lift_values_invalid(sample_mmmdata, sample_calibration_input):
    new_calibration_input = create_modified_calibration_input(
        sample_calibration_input, ("radio_spend",), lift_abs="invalid"  # Changed to tuple
    )

    validator = CalibrationInputValidation(
        sample_mmmdata, new_calibration_input, window_start=datetime(2022, 1, 1), window_end=datetime(2022, 1, 10)
    )
    result = validator._check_lift_values()
    assert result.status == False
    assert ("radio_spend",) in result.error_details
    assert "must be a valid number" in result.error_message


def test_check_spend_values_invalid(sample_mmmdata, sample_calibration_input):
    new_calibration_input = create_modified_calibration_input(
        sample_calibration_input, ("tv_spend",), spend=1000  # Changed to tuple
    )

    validator = CalibrationInputValidation(
        sample_mmmdata, new_calibration_input, window_start=datetime(2022, 1, 1), window_end=datetime(2022, 1, 10)
    )
    result = validator._check_spend_values()
    assert result.status == False
    assert ("tv_spend",) in result.error_details
    assert "does not match the input data" in result.error_message


def test_check_confidence_values_invalid(sample_mmmdata, sample_calibration_input):
    new_calibration_input = create_modified_calibration_input(
        sample_calibration_input, ("radio_spend",), confidence=0.7  # Changed to tuple
    )

    validator = CalibrationInputValidation(
        sample_mmmdata, new_calibration_input, window_start=datetime(2022, 1, 1), window_end=datetime(2022, 1, 10)
    )
    result = validator._check_confidence_values()
    assert result.status == False
    assert ("radio_spend",) in result.error_details
    assert "lower than 80%" in result.error_message


def test_check_metric_values_invalid(sample_mmmdata, sample_calibration_input):
    new_calibration_input = create_modified_calibration_input(
        sample_calibration_input, ("tv_spend",), metric=DependentVarType.CONVERSION  # Changed to tuple
    )

    validator = CalibrationInputValidation(
        sample_mmmdata, new_calibration_input, window_start=datetime(2022, 1, 1), window_end=datetime(2022, 1, 10)
    )
    result = validator._check_metric_values()
    assert result.status == False
    assert ("tv_spend",) in result.error_details
    assert "does not match the dependent variable" in result.error_message


def test_check_obj_weights_valid(sample_mmmdata, sample_calibration_input):
    validator = CalibrationInputValidation(
        sample_mmmdata, sample_calibration_input, window_start=datetime(2022, 1, 1), window_end=datetime(2022, 1, 10)
    )
    result = validator.check_obj_weights([0, 1, 1], True)
    assert result.status == True
    assert not result.error_details
    assert not result.error_message


def test_check_obj_weights_invalid(sample_mmmdata, sample_calibration_input):
    validator = CalibrationInputValidation(
        sample_mmmdata, sample_calibration_input, window_start=datetime(2022, 1, 1), window_end=datetime(2022, 1, 10)
    )

    # Test case 1: Incorrect number of weights
    result = validator.check_obj_weights([0, 1, 1, 1], False)
    assert result.status == False
    assert "length" in result.error_details
    assert "Invalid number of objective weights" in result.error_message

    # Test case 2: Weights out of valid range
    result = validator.check_obj_weights([-1, 1, 11], False)
    assert result.status == False
    assert "range" in result.error_details
    assert "Objective weights out of valid range" in result.error_message


def test_validate(sample_mmmdata, sample_calibration_input):
    # Test with valid input
    validator = CalibrationInputValidation(
        sample_mmmdata, sample_calibration_input, window_start=datetime(2022, 1, 1), window_end=datetime(2022, 1, 10)
    )

    results = validator.validate()
    assert len(results) == 1
    assert all(result.status for result in results)
    assert all(not result.error_details for result in results)
    assert all(not result.error_message for result in results)

    # Test with invalid input
    invalid_calibration_input = create_modified_calibration_input(
        sample_calibration_input,
        ("tv_spend",),  # Changed to tuple
        lift_start_date=datetime(2021, 12, 31),
        lift_abs="invalid",
        spend=1000000,
        confidence=0.5,
        metric=DependentVarType.CONVERSION,
    )

    invalid_validator = CalibrationInputValidation(
        sample_mmmdata, invalid_calibration_input, window_start=datetime(2022, 1, 1), window_end=datetime(2022, 1, 10)
    )
    invalid_results = invalid_validator.validate()
    assert len(invalid_results) == 1
    assert any(not result.status for result in invalid_results)
    assert any(result.error_details for result in invalid_results)
    assert any(result.error_message for result in invalid_results)


def test_multichannel_validation(sample_mmmdata, sample_multichannel_calibration_input):
    validator = CalibrationInputValidation(
        sample_mmmdata,
        sample_multichannel_calibration_input,
        window_start=datetime(2022, 1, 1),
        window_end=datetime(2022, 1, 10),
    )
    result = validator.check_calibration()
    assert result.status == True
    assert not result.error_details
    assert not result.error_message


def test_invalid_channel(sample_mmmdata, sample_calibration_input):
    # Test with non-existent channel
    invalid_input = create_modified_calibration_input(
        sample_calibration_input, ("nonexistent_channel",), lift_abs=1000
    )

    validator = CalibrationInputValidation(
        sample_mmmdata, invalid_input, window_start=datetime(2022, 1, 1), window_end=datetime(2022, 1, 10)
    )
    result = validator._check_spend_values()
    assert result.status == False
    assert ("nonexistent_channel",) in result.error_details
    assert "not found in data" in result.error_message.lower()


def test_invalid_multichannel_combination(sample_mmmdata, sample_calibration_input):
    # Test with one valid and one invalid channel
    invalid_combination = CalibrationInput(
        channel_data={
            ("tv_spend", "nonexistent_channel"): ChannelCalibrationData(
                lift_start_date=datetime(2022, 1, 1),
                lift_end_date=datetime(2022, 1, 5),
                lift_abs=1000,
                spend=300,
                confidence=0.9,
                metric=DependentVarType.REVENUE,
                calibration_scope=CalibrationScope.IMMEDIATE,
            )
        }
    )

    validator = CalibrationInputValidation(
        sample_mmmdata, invalid_combination, window_start=datetime(2022, 1, 1), window_end=datetime(2022, 1, 10)
    )
    result = validator._check_spend_values()
    assert result.status == False
    assert ("tv_spend", "nonexistent_channel") in result.error_details
    assert "not found in data" in result.error_message.lower()


def test_edge_cases(sample_mmmdata):
    # Test with empty calibration input
    empty_input = CalibrationInput(channel_data={})
    validator = CalibrationInputValidation(
        sample_mmmdata, empty_input, window_start=datetime(2022, 1, 1), window_end=datetime(2022, 1, 10)
    )
    result = validator.check_calibration()
    assert result.status == True  # Empty input should be valid

    # Test with None calibration input
    validator_none = CalibrationInputValidation(
        sample_mmmdata, None, window_start=datetime(2022, 1, 1), window_end=datetime(2022, 1, 10)
    )
    result_none = validator_none.check_calibration()
    assert result_none.status == True  # None input should be valid


def test_date_boundary_cases(sample_mmmdata, sample_calibration_input):
    # Test exact boundary dates
    boundary_input = create_modified_calibration_input(
        sample_calibration_input,
        ("tv_spend",),
        lift_start_date=datetime(2022, 1, 1),  # Exact start
        lift_end_date=datetime(2022, 1, 10),  # Exact end
    )

    validator = CalibrationInputValidation(
        sample_mmmdata, boundary_input, window_start=datetime(2022, 1, 1), window_end=datetime(2022, 1, 10)
    )
    result = validator._check_date_range()
    assert result.status == True
    assert not result.error_details


# Modify your existing test_validate to include multi-channel cases
def test_validate_with_multichannel(sample_mmmdata, sample_multichannel_calibration_input):
    validator = CalibrationInputValidation(
        sample_mmmdata,
        sample_multichannel_calibration_input,
        window_start=datetime(2022, 1, 1),
        window_end=datetime(2022, 1, 10),
    )

    results = validator.validate()
    assert len(results) == 1
    assert all(result.status for result in results)
    assert all(not result.error_details for result in results)
    assert all(not result.error_message for result in results)
