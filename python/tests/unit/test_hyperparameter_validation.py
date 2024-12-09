import pytest
from src.robyn.data.entities.hyperparameters import (
    Hyperparameters,
    ChannelHyperparameters,
)
from src.robyn.data.validation.hyperparameter_validation import (
    HyperparametersValidation,
)
from src.robyn.data.entities.enums import AdstockType
from src.robyn.data.validation.validation import ValidationResult


@pytest.fixture
def sample_hyperparameters():
    return Hyperparameters(
        hyperparameters={
            "channel1": ChannelHyperparameters(
                thetas=[0.1, 0.9],
                alphas=[0.5, 3.0],
                gammas=[0.3, 1.0],
                shapes=[0.0001, 2.0],
                scales=[0.0001, 1.0],
            ),
            "channel2": ChannelHyperparameters(
                thetas=[0.1, 0.9],
                alphas=[0.5, 3.0],
                gammas=[0.3, 1.0],
                shapes=[0.0001, 2.0],
                scales=[0.0001, 1.0],
            ),
        },
        adstock=AdstockType.GEOMETRIC,
        train_size=[0.6, 0.9],
    )


def test_check_hyperparameters_no_issues(sample_hyperparameters):
    validator = HyperparametersValidation(sample_hyperparameters)
    result = validator.check_hyperparameters()
    assert result.status == True
    assert not result.error_details
    assert not result.error_message


def test_check_train_size_valid(sample_hyperparameters):
    validator = HyperparametersValidation(sample_hyperparameters)
    validator.check_train_size()  # Should not raise an exception


def test_check_train_size_invalid(sample_hyperparameters):
    sample_hyperparameters.train_size = [0.05, 1.1]
    validator = HyperparametersValidation(sample_hyperparameters)
    with pytest.raises(ValueError):
        validator.check_train_size()


def test_hyper_names_geometric(sample_hyperparameters):
    validator = HyperparametersValidation(sample_hyperparameters)
    all_media = sample_hyperparameters.hyperparameters.keys()
    names = validator.hyper_names(all_media=all_media)
    assert set(names) == set(
        [
            "channel1_thetas",
            "channel2_thetas",
            "channel1_alphas",
            "channel2_alphas",
            "channel1_gammas",
            "channel2_gammas",
            "lambda",
            "train_size",
        ]
    )


def test_hyper_names_weibull(sample_hyperparameters):
    sample_hyperparameters.adstock = AdstockType.WEIBULL_CDF
    validator = HyperparametersValidation(sample_hyperparameters)
    all_media = sample_hyperparameters.hyperparameters.keys()
    names = validator.hyper_names(all_media=all_media)
    assert set(names) == set(
        [
            "channel1_shapes",
            "channel2_shapes",
            "channel1_scales",
            "channel2_scales",
            "channel1_alphas",
            "channel2_alphas",
            "channel1_gammas",
            "channel2_gammas",
            "lambda",
            "train_size",
        ]
    )


def test_check_hyper_limits_valid(sample_hyperparameters):
    validator = HyperparametersValidation(sample_hyperparameters)
    validator.check_hyper_limits("thetas")  # Should not raise an exception


def test_check_hyper_limits_invalid(sample_hyperparameters):
    sample_hyperparameters.hyperparameters["channel1"].thetas = [-0.1, 1.1]
    validator = HyperparametersValidation(sample_hyperparameters)
    with pytest.raises(ValueError):
        validator.check_hyper_limits("thetas")


def test_validate_all_checks(sample_hyperparameters):
    validator = HyperparametersValidation(sample_hyperparameters)
    results = validator.validate()
    assert all(result.status for result in results)


def test_validate_with_issues(sample_hyperparameters):
    sample_hyperparameters.train_size = [0.05, 1.1]
    sample_hyperparameters.hyperparameters["channel1"].alphas = [-1, 11]
    validator = HyperparametersValidation(sample_hyperparameters)
    results = validator.validate()
    print("The results: ")
    print(results)
    assert not all(result.status for result in results)
    assert any("train_size" in result.error_details for result in results)
    assert any("hyperparameters" in result.error_details for result in results)
