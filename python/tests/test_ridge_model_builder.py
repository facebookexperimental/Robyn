# test_ridge_model_builder.py

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from robyn.modeling.ridge_model_builder import RidgeModelBuilder
from robyn.data.entities.mmmdata import MMMData
from robyn.data.entities.holidays_data import HolidaysData
from robyn.data.entities.calibration_input import CalibrationInput
from robyn.data.entities.hyperparameters import Hyperparameters
from robyn.modeling.feature_engineering import FeaturizedMMMData
from robyn.modeling.entities.enums import NevergradAlgorithm
from robyn.data.entities.enums import AdstockType
from robyn.modeling.entities.modelrun_trials_config import TrialsConfig
from robyn.modeling.entities.modeloutputs import ModelOutputs, Trial
import pytest
import pandas as pd
import numpy as np
from robyn.data.entities.mmmdata import MMMData
from robyn.data.entities.enums import DependentVarType, PaidMediaSigns, OrganicSigns, ContextSigns
from robyn.data.entities.holidays_data import HolidaysData
from robyn.data.entities.enums import ProphetVariableType, ProphetSigns


@pytest.fixture
def sample_mmmdata():
    data = pd.DataFrame(
        {
            "date": pd.date_range(start="2020-01-01", periods=100),
            "sales": np.random.rand(100),
            "tv_spend": np.random.rand(100),
            "radio_spend": np.random.rand(100),
            "newspaper_spend": np.random.rand(100),
        }
    )
    # Fill NaN values with the mean of each column
    data.fillna(data.mean(), inplace=True)
    mmmdata_spec = MMMData.MMMDataSpec(
        dep_var="sales",
        dep_var_type=DependentVarType.REVENUE,
        date_var="date",
        paid_media_spends=["tv_spend", "radio_spend", "newspaper_spend"],
        paid_media_vars=None,
        paid_media_signs=[PaidMediaSigns.POSITIVE] * 3,
        organic_vars=None,
        organic_signs=None,
        context_vars=None,
        context_signs=None,
        factor_vars=None,
    )
    return MMMData(data=data, mmmdata_spec=mmmdata_spec)


@pytest.fixture
def sample_holidays_data():
    # Create a DataFrame with the necessary holiday data
    dt_holidays = pd.DataFrame(
        {"date": pd.date_range(start="2020-01-01", periods=100), "holiday": np.random.choice([0, 1], size=100)}
    )
    # Assuming example values for prophet_vars and prophet_signs
    prophet_vars = [ProphetVariableType.HOLIDAY] * 3  # Adjust based on actual usage
    prophet_signs = [ProphetSigns.POSITIVE] * 3  # Adjust based on actual usage
    prophet_country = "US"  # Example country, adjust as necessary
    # Return an instance of HolidaysData
    return HolidaysData(
        dt_holidays=dt_holidays,
        prophet_vars=prophet_vars,
        prophet_signs=prophet_signs,
        prophet_country=prophet_country,
    )


@pytest.fixture
def sample_hyperparameters():
    return Hyperparameters(
        hyperparameters={
            "tv_spend_thetas": [0.1, 0.9],
            "radio_spend_thetas": [0.1, 0.9],
            "newspaper_spend_thetas": [0.1, 0.9],
            "tv_spend_alphas": [0.5, 2.0],
            "radio_spend_alphas": [0.5, 2.0],
            "newspaper_spend_alphas": [0.5, 2.0],
            "tv_spend_gammas": [0.3, 1.0],
            "radio_spend_gammas": [0.3, 1.0],
            "newspaper_spend_gammas": [0.3, 1.0],
        },
        lambda_=[0.1, 10],
    )


@pytest.fixture
def sample_featurized_mmm_data(sample_mmmdata):
    return FeaturizedMMMData(dt_mod=sample_mmmdata.data, dt_modRollWind=sample_mmmdata.data, modNLS={})


@pytest.fixture
def ridge_model_builder(sample_mmmdata, sample_holidays_data, sample_hyperparameters, sample_featurized_mmm_data):
    return RidgeModelBuilder(
        sample_mmmdata, sample_holidays_data, CalibrationInput(), sample_hyperparameters, sample_featurized_mmm_data
    )


def test_ridge_model_builder_initialization(ridge_model_builder):
    assert isinstance(ridge_model_builder, RidgeModelBuilder)
    assert isinstance(ridge_model_builder.mmm_data, MMMData)
    assert isinstance(ridge_model_builder.holidays_data, HolidaysData)
    assert isinstance(ridge_model_builder.hyperparameters, Hyperparameters)
    assert isinstance(ridge_model_builder.featurized_mmm_data, FeaturizedMMMData)


def test_build_models(ridge_model_builder):
    trials_config = TrialsConfig(trials=1, iterations=10)
    result = ridge_model_builder.build_models(trials_config)
    print("Type of result.trials:", type(result.trials))
    print("Content of result.trials:", result.trials)
    assert isinstance(result, ModelOutputs)
    assert len(result.trials) == 10
    assert result.iterations == 10


@pytest.mark.parametrize("adstock_type", [AdstockType.GEOMETRIC, AdstockType.WEIBULL_CDF, AdstockType.WEIBULL_PDF])
def test_prepare_data(ridge_model_builder, adstock_type):
    ridge_model_builder.mmm_data.adstock = adstock_type
    hyperparams = {
        "tv_spend_thetas": 0.5,
        "radio_spend_thetas": 0.6,
        "newspaper_spend_thetas": 0.7,
        "tv_spend_alphas": 1.0,
        "radio_spend_alphas": 1.1,
        "newspaper_spend_alphas": 1.2,
        "tv_spend_gammas": 0.5,
        "radio_spend_gammas": 0.6,
        "newspaper_spend_gammas": 0.7,
        "tv_spend_shapes": 1.5,
        "radio_spend_shapes": 1.6,
        "newspaper_spend_shapes": 1.7,
        "tv_spend_scales": 2.0,
        "radio_spend_scales": 2.1,
        "newspaper_spend_scales": 2.2,
    }
    X, y = ridge_model_builder._prepare_data(hyperparams, adstock_type)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert "tv_spend" in X.columns
    assert "radio_spend" in X.columns
    assert "newspaper_spend" in X.columns
    assert len(X) == len(y)


def test_geometric_adstock():
    x = pd.Series([1, 2, 3, 4, 5])
    theta = 0.5
    result = RidgeModelBuilder._geometric_adstock(x, theta)
    assert isinstance(result, pd.Series)
    assert len(result) == len(x)
    assert result.iloc[0] == 1
    assert result.iloc[1] == 2 + 0.5 * 1


def test_weibull_adstock():
    x = pd.Series([1, 2, 3, 4, 5])
    shape = 1.5
    scale = 2.0
    result_cdf = RidgeModelBuilder._weibull_adstock(x, shape, scale, AdstockType.WEIBULL_CDF)
    result_pdf = RidgeModelBuilder._weibull_adstock(x, shape, scale, AdstockType.WEIBULL_PDF)
    assert isinstance(result_cdf, pd.Series)
    assert isinstance(result_pdf, pd.Series)
    assert len(result_cdf) == len(x)
    assert len(result_pdf) == len(x)


def test_hill_transformation():
    x = pd.Series([1, 2, 3, 4, 5])
    alpha = 1.0
    gamma = 0.5
    result = RidgeModelBuilder._hill_transformation(x, alpha, gamma)
    assert isinstance(result, pd.Series)
    assert len(result) == len(x)
    assert all(0 <= val <= 1 for val in result)


def test_calculate_rssd():
    coefs = np.array([1, 2, 3, 0, 0])
    rssd_with_penalty = RidgeModelBuilder._calculate_rssd(coefs, rssd_zero_penalty=True)
    rssd_without_penalty = RidgeModelBuilder._calculate_rssd(coefs, rssd_zero_penalty=False)
    assert rssd_with_penalty > rssd_without_penalty


def test_calculate_objective():
    rsq_train = 0.8
    rsq_test = 0.7
    rssd = 0.5
    objective_weights = {"rsq_train": 1.0, "rsq_test": 0.5, "rssd": 2.0}
    result = RidgeModelBuilder._calculate_objective(rsq_train, rsq_test, rssd, objective_weights)
    assert isinstance(result, float)
    assert result > 0


@patch("robyn.modeling.ridge_model_builder.ng.optimizers.registry")
def test_run_nevergrad_optimization(mock_ng_registry, ridge_model_builder):
    mock_optimizer = Mock()
    mock_ng_registry.__getitem__.return_value.return_value = mock_optimizer
    mock_optimizer.ask.return_value = Mock(value=np.array([0.5] * 10))  # Increase the number of values

    hyper_collect = ridge_model_builder._hyper_collector(
        adstock="geometric",
        all_media=["tv_spend", "radio_spend", "newspaper_spend"],
        paid_media_spends=["tv_spend", "radio_spend", "newspaper_spend"],
        organic_vars=[],
        prophet_vars=[],
        context_vars=[],
        dt_mod=ridge_model_builder.featurized_mmm_data.dt_mod,
        hyper_in=ridge_model_builder.hyperparameters,
        ts_validation=True,
        add_penalty_factor=False,
    )

    result = ridge_model_builder.run_nevergrad_optimization(
        hyper_collect=hyper_collect, iterations=10, cores=1, nevergrad_algo=NevergradAlgorithm.TWO_POINTS_DE
    )

    assert isinstance(result, list)
    assert len(result) == 10
    assert all(isinstance(item, Trial) for item in result)


if __name__ == "__main__":
    pytest.main()
