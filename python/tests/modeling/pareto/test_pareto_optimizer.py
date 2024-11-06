import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
from robyn.data.entities.mmmdata import MMMData
from robyn.modeling.entities.modeloutputs import ModelOutputs
from robyn.modeling.pareto.response_curve import ResponseCurveCalculator, ResponseOutput
from robyn.modeling.pareto.immediate_carryover import ImmediateCarryoverCalculator
from robyn.modeling.pareto.pareto_utils import ParetoUtils
from robyn.data.entities.hyperparameters import ChannelHyperparameters, Hyperparameters
from robyn.modeling.feature_engineering import FeaturizedMMMData
from robyn.data.entities.holidays_data import HolidaysData
from robyn.data.entities.enums import AdstockType
from robyn.modeling.transformations.transformations import Transformation, AdstockResult
from robyn.modeling.pareto.pareto_optimizer import (
    ParetoOptimizer,
    ParetoResult,
    ParetoData,
) 

@pytest.fixture
def pareto_data_data_factory():
    def _create_pareto_data(aggregated_data = None):
        decomp_spend_dist_df = pd.DataFrame({
            'solID': ['test'],
            'rn': ['media'],
        })
        dt_hyppar = pd.DataFrame({
            'solID': ['test'],
            'media_alphas': [1],
            'media_gammas': [2],
        })
        pareto_data = ParetoData(
            decomp_spend_dist=decomp_spend_dist_df, 
            x_decomp_agg=aggregated_data["x_decomp_agg"] if aggregated_data is not None else [], 
            result_hyp_param=dt_hyppar, 
            pareto_fronts=[]
        )
        return pareto_data
    return _create_pareto_data

@pytest.fixture
def trial_mock_data_factory():
    def _create_trial_mock():
        trial_mock = MagicMock()
        trial_mock.decomp_spend_dist = pd.DataFrame({
            "trial": [1, 2, 3],
            "iterNG": [1, 1, 1],
            "iterPar": [1, 2, 3],
            "solID": ["sol1", "sol2", "sol3"]
        })
        return trial_mock
    return _create_trial_mock
    
@pytest.fixture
def aggregated_data_factory():
    def _create_aggregated_data():
        return {
            "result_hyp_param": pd.DataFrame({
                "mape": [1, 2, 3],
                "nrmse": [1, 2, 3],
                "decomp.rssd": [1, 2, 3],
                "nrmse_train": [1, 2, 3],
                "iterNG": [1, 2, 3],
                "iterPar": [1, 2, 3],
                "solID": ["sol1", "sol2", "sol3"],
                "robynPareto": [1, 2, 3],
            }),
            "x_decomp_agg": pd.DataFrame({
                "rn": ["media", "media", "media"],
                "solID": ["test", "test2", "test3"],
                "coef": [1, 0.5, 0.5]
            }),
            "result_calibration": None,
        }
    return _create_aggregated_data

@pytest.fixture
def setup_optimizer():
    # Mocking dependencies
    mmmdata_spec = MMMData.MMMDataSpec(
        paid_media_spends=["media"],
        paid_media_vars=["media"],
        organic_vars=[],
        date_var="date",
        rolling_window_start_which=0,
        rolling_window_end_which=10
    )
    # Create a real DataFrame for MMMData
    data = pd.DataFrame({
        "date": pd.date_range(start="2020-01-01", periods=20, freq='D'),
        "media": np.random.rand(20)  # Random data for the 'media' column
    })
    # Create a real MMMData instance
    mmm_data = MMMData(data=data, mmmdata_spec=mmmdata_spec)
    model_outputs = MagicMock(spec=ModelOutputs)
    hyper_parameter = MagicMock(spec=Hyperparameters)
    featurized_mmm_data = type('', (), {})()
    featurized_mmm_data.dt_mod = pd.DataFrame({
        'media': [0.5] * 365,
        'dep_var': [1.0] * 365 
    })
    holidays_data = MagicMock(spec=HolidaysData)
    # Instance of ParetoOptimizer
    optimizer = ParetoOptimizer(
        mmm_data=mmm_data,
        model_outputs=model_outputs,
        hyper_parameter=hyper_parameter,
        featurized_mmm_data=featurized_mmm_data,
        holidays_data=holidays_data,
    )
    optimizer._compute_response_curves = MagicMock(
        return_value=ParetoData(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), [])
    )
    optimizer._generate_plot_data = MagicMock(
        return_value={
            "pareto_solutions": [],
            "mediaVecCollect": pd.DataFrame(),
            "xDecompVecCollect": pd.DataFrame(),
            "plotDataCollect": {},
            "df_caov_pct_all": pd.DataFrame(),
        }
    )
    return optimizer

def test_optimize(setup_optimizer, pareto_data_data_factory):
    optimizer = setup_optimizer
    # Setup mock return values for methods called within optimize
    optimizer._aggregate_model_data = MagicMock(
        return_value={
            "result_hyp_param": pd.DataFrame(),
            "result_calibration": pd.DataFrame(),
        }
    )
    optimizer._compute_pareto_fronts = MagicMock(return_value=pd.DataFrame())
    optimizer.prepare_pareto_data = MagicMock(
        return_value=pareto_data_data_factory()
    )
    # Run the optimize function
    result = optimizer.optimize()
    # Assertions to check function calls and return values
    optimizer._aggregate_model_data.assert_called_once()
    optimizer._compute_pareto_fronts.assert_called_once()
    optimizer.prepare_pareto_data.assert_called_once()
    optimizer._compute_response_curves.assert_called_once()
    optimizer._generate_plot_data.assert_called_once()
    assert isinstance(result, ParetoResult)

def test_aggregate_model_data(setup_optimizer):
    optimizer = setup_optimizer
    # Setup mock return values
    optimizer.model_outputs.hyper_fixed = True
    optimizer.model_outputs.trials = [
        MagicMock(
            result_hyp_param=pd.DataFrame(), 
            x_decomp_agg=pd.DataFrame({'solID': [1]}),
        )
    ]
    # Run the _aggregate_model_data function
    result = optimizer._aggregate_model_data(calibrated=False)
    # Assertions to check the return value
    assert isinstance(result, dict)
    assert "result_hyp_param" in result
    assert "x_decomp_agg" in result

@patch('robyn.modeling.pareto.pareto_optimizer.ParetoOptimizer._pareto_fronts')
def test_compute_pareto_fronts_hyper_fixed_false(mock_pareto_fronts, setup_optimizer, aggregated_data_factory):
    optimizer = setup_optimizer
    # Setup mock data
    mock_pareto_fronts.return_value = pd.DataFrame({
        'x': [], 
        'y': [], 
        'pareto_front': []
    })
    optimizer.model_outputs.hyper_fixed = False 
    optimizer.model_outputs.ts_validation = None
    # Run the _compute_pareto_fronts function
    result = optimizer._compute_pareto_fronts(
        aggregated_data=aggregated_data_factory(),
        pareto_fronts="auto",
        min_candidates=100,
        calibration_constraint=0.1,
    )
    # Assertions to check the return value
    assert isinstance(result, pd.DataFrame)

@patch('robyn.modeling.pareto.pareto_optimizer.ParetoOptimizer._pareto_fronts')
def test_compute_pareto_fronts_hyper_fixed_true(mock_pareto_fronts, setup_optimizer, aggregated_data_factory):
    optimizer = setup_optimizer
    # Setup mock data
    mock_pareto_fronts.return_value = pd.DataFrame({
        'x': [], 
        'y': [], 
        'pareto_front': []
    })
    optimizer.model_outputs.hyper_fixed = True
    optimizer.model_outputs.ts_validation = None
    # Run the _compute_pareto_fronts function
    result = optimizer._compute_pareto_fronts(
        aggregated_data=aggregated_data_factory(),
        pareto_fronts="auto",
        min_candidates=100,
        calibration_constraint=0.1,
    )
    # Assertions to check the return value
    assert isinstance(result, pd.DataFrame)

def test_prepare_pareto_data_hyper_fixed_true(setup_optimizer, aggregated_data_factory, trial_mock_data_factory):
    optimizer = setup_optimizer
    # Setup mock data
    optimizer.model_outputs.trials = [trial_mock_data_factory()]
    optimizer.model_outputs.hyper_fixed = True
    # Run the prepare_pareto_data function
    result = optimizer.prepare_pareto_data(
        aggregated_data=aggregated_data_factory(),
        pareto_fronts="auto",
        min_candidates=100,
        calibrated=False,
    )
    # Assertions to check the return value
    assert isinstance(result, ParetoData)

def test_prepare_pareto_data_hyper_fixed_false(setup_optimizer, aggregated_data_factory, trial_mock_data_factory):
    optimizer = setup_optimizer
    # Setup mock data
    optimizer.model_outputs.trials = [trial_mock_data_factory()]
    optimizer.model_outputs.hyper_fixed = False
    # Run the prepare_pareto_data function

    result = optimizer.prepare_pareto_data(
        aggregated_data=aggregated_data_factory(),
        pareto_fronts="auto",
        min_candidates=2,
        calibrated=False,
    )
    # Assertions to check the return value
    assert isinstance(result, ParetoData)

@patch('robyn.modeling.transformations.transformations.Transformation.transform_adstock')
def test_run_dt_resp(mock_transform_adstock, setup_optimizer, aggregated_data_factory, pareto_data_data_factory):
    optimizer = setup_optimizer
    # Setup mock data
    row = pd.Series({"solID": "test", "rn": "media", "mean_spend": 1})
    aggregated_data = aggregated_data_factory()
    adstock_result = MagicMock(spec=AdstockResult)
    adstock_result.x = pd.Series([1, 2, 3]) 
    adstock_result.x_decayed = pd.Series([4, 5, 6])  
    adstock_result.x_imme = pd.Series([7, 8, 9])  
    mock_transform_adstock.return_value = adstock_result
    # Run the run_dt_resp function
    result = optimizer.run_dt_resp(row=row, paretoData=pareto_data_data_factory(aggregated_data))
    # Assertions to check the return value
    expected_result = pd.Series({
        "mean_response": 0.333333,
        "mean_spend_adstocked": 5.0,
        "mean_carryover": 3.0,
        "rn": "media",
        "solID": "test"
    })
    pd.testing.assert_series_equal(result, expected_result)
    assert isinstance(result, pd.Series)

def test_generate_plot_data(setup_optimizer, aggregated_data_factory, pareto_data_data_factory):
    optimizer = setup_optimizer
    # Setup mock data
    optimizer.featurized_mmm_data.dt_modRollWind = pd.DataFrame({
        'ds': pd.date_range(start='2023-01-01', periods=10, freq='D')
    })
    # Run the _generate_plot_data function
    result = optimizer._generate_plot_data(
        aggregated_data=aggregated_data_factory(), pareto_data=pareto_data_data_factory()
    )
    # Assertions to check the return value
    assert isinstance(result, dict)
    assert "pareto_solutions" in result
    assert "mediaVecCollect" in result
    assert "xDecompVecCollect" in result
    assert "plotDataCollect" in result
    assert "df_caov_pct_all" in result

def test_robyn_immcarr(setup_optimizer, aggregated_data_factory, pareto_data_data_factory):
    optimizer = setup_optimizer
    # Setup mock data
    aggregated_data = aggregated_data_factory()

    optimizer.mmm_data.mmmdata_spec.window_start = '2023-01-01'  
    optimizer.mmm_data.mmmdata_spec.window_end = '2023-01-10'
    optimizer.mmm_data.mmmdata_spec.all_media = ['media']
    optimizer.featurized_mmm_data.dt_modRollWind = pd.DataFrame({
        'ds': pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    })
    optimizer.hyper_parameter.adstock = AdstockType.GEOMETRIC
    result_hyp_param = pd.DataFrame({
        'solID': ['test'],
        'media_alphas': [0.1],
        'media_gammas': [0.2],
        'media_thetas': [0.3]  # Include this if adstock is GEOMETRIC
    })
    # Run the robyn_immcarr function
    result = optimizer.robyn_immcarr(
        pareto_data=pareto_data_data_factory(aggregated_data), result_hyp_param=result_hyp_param
    )
    # Assertions to check the return value
    assert isinstance(result, pd.DataFrame)
    expected_result = pd.DataFrame({
        "solID": ["test", "test"],
        "start_date": ["2023-01-01", "2023-01-01"],
        "end_date": ["2023-01-10", "2023-01-10"],
        "rn": ["media", "media"],
        "type": ["Carryover", "Immediate"],
        "response": [4.278791, 0.777566],
        "percentage": [1.0, 1.0],  # Ensure these are floats if that's the expected type
        "carryover_pct": [0.84622, 0.84622]
    })
    pd.testing.assert_frame_equal(result, expected_result)

def test_extract_hyperparameter(setup_optimizer):
    optimizer = setup_optimizer
    # Setup mock data
    hyp_param_sam = pd.DataFrame({
        'media_alphas': [0.1, 0.2],
        'media_gammas': [0.3, 0.4],
        'media_thetas': [0.5, 0.6],  
        'media_shapes': [0.7, 0.8],  
        'media_scales': [0.9, 1.0],  
    })
    # Run the _extract_hyperparameter function
    result = optimizer._extract_hyperparameter(hypParamSam=hyp_param_sam)
    # Assertions to check the return value
    assert isinstance(result, Hyperparameters)

def test_model_decomp(setup_optimizer):
    optimizer = setup_optimizer
    # Setup mock data
    inputs = {
        "coefs": pd.DataFrame({"name": ["intercept", "feature1"], "coefficient": [1.0, 0.5]}),
        "y_pred": pd.Series([1.5, 2.0, 2.5]),
        "dt_modSaturated": pd.DataFrame({
            "dep_var": [1.0, 2.0, 3.0],
            "feature1": [0.5, 1.0, 1.5]
        }),
        "dt_saturatedImmediate": pd.DataFrame({"feature1": [0.1, 0.2, 0.3]}),
        "dt_saturatedCarryover": pd.DataFrame({"feature1": [0.05, 0.1, 0.15]}),
        "dt_modRollWind": pd.DataFrame({"ds": ["2023-01-01", "2023-01-02", "2023-01-03"]}),
        "refreshAddedStart": None,
    }
    # Run the _model_decomp function
    result = optimizer._model_decomp(inputs=inputs)
    # Assertions to check the return value
    assert isinstance(result, dict)
    assert "xDecompVec" in result
    assert "mediaDecompImmediate" in result
    assert "mediaDecompCarryover" in result
