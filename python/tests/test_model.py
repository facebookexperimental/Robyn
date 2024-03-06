import pandas as pd
import numpy as np
import re
import pytest

from python import model
@pytest.mark.xfail(raises=NotImplementedError)
def test_model_decomp():
    # Create dummy data
    coefs = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    dt_modSaturated = pd.DataFrame({'dep_var': [1, 2, 3, 4, 5],
                                    'col1': [0.1, 0.2, 0.3, 0.4, 0.5],
                                    'col2': [0.5, 0.4, 0.3, 0.2, 0.1]})
    dt_saturatedImmediate = pd.DataFrame({'season': [1, 2, 3, 4, 5],
                                          'competitor_sales_B': [0.1, 0.2, 0.3, 0.4, 0.5]})
    dt_saturatedCarryover = pd.DataFrame({'season': [1, 2, 3, 4, 5],
                                          'competitor_sales_B': [0.5, 0.4, 0.3, 0.2, 0.1]})
    dt_modRollWind = pd.DataFrame({'ds': [1, 2, 3, 4, 5]})
    refreshAddedStart = 1

    # Call the function
    result = model.model_decomp(coefs, y_pred, dt_modSaturated, dt_saturatedImmediate,
                          dt_saturatedCarryover, dt_modRollWind, refreshAddedStart)

    # Perform assertions on the output
    assert isinstance(result, dict)
    assert 'xDecompVec' in result
    assert 'xDecompVec.scaled' in result
    assert 'xDecompAgg' in result
    assert 'coefsOutCat' in result
    assert 'mediaDecompImmediate' in result
    assert 'mediaDecompCarryover' in result

    assert isinstance(result['xDecompVec'], pd.DataFrame)
    assert isinstance(result['xDecompVec.scaled'], pd.DataFrame)
    assert isinstance(result['xDecompAgg'], pd.DataFrame)
    assert isinstance(result['coefsOutCat'], pd.DataFrame)
    assert isinstance(result['mediaDecompImmediate'], pd.DataFrame)
    assert isinstance(result['mediaDecompCarryover'], pd.DataFrame)

    assert result['xDecompVec'].shape[0] == dt_modSaturated.shape[0]
    assert result['xDecompVec.scaled'].shape[0] == dt_modSaturated.shape[0]
    assert result['xDecompAgg'].shape[0] == len(coefs) - 1
    assert result['coefsOutCat'].shape[0] == len(coefs)
    assert result['mediaDecompImmediate'].shape[0] == dt_saturatedImmediate.shape[0]
    assert result['mediaDecompCarryover'].shape[0] == dt_saturatedCarryover.shape[0]

    assert 'xDecompAgg' in result['xDecompAgg']
    assert 'xDecompPerc' in result['xDecompAgg']
    assert 'xDecompMeanNon0' in result['xDecompAgg']
    assert 'xDecompMeanNon0Perc' in result['xDecompAgg']
    assert 'xDecompAggRF' in result['xDecompAgg']
    assert 'xDecompPercRF' in result['xDecompAgg']
    assert 'xDecompMeanNon0RF' in result['xDecompAgg']
    assert 'xDecompMeanNon0PercRF' in result['xDecompAgg']
    assert 'pos' in result['xDecompAgg']

    assert 'rn' in result['coefsOutCat']
    assert 'coefs' in result['coefsOutCat']

    assert 'ds' in result['mediaDecompImmediate']
    assert 'y' in result['mediaDecompImmediate']

    assert 'ds' in result['mediaDecompCarryover']
    assert 'y' in result['mediaDecompCarryover']

    # Add more assertions as needed

@pytest.mark.xfail(raises=NotImplementedError)
def test_robyn_run_outputs():
    # Create dummy data
    InputCollect = {
        'hyperparameters': {
            'param1': [1, 2, 3],
            'param2': [4, 5, 6]
        },
        'calibration_input': pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
    }
    dt_hyper_fixed = None
    json_file = None
    ts_validation = False
    add_penalty_factor = False
    refresh = False
    seed = 123
    quiet = False
    cores = None
    trials = 5
    iterations = 2000
    rssd_zero_penalty = True
    objective_weights = None
    nevergrad_algo = "TwoPointsDE"
    intercept = True
    intercept_sign = "non_negative"
    lambda_control = None
    outputs = True

    # Call the function
    result = model.robyn_run(InputCollect, dt_hyper_fixed, json_file, ts_validation,
                             add_penalty_factor, refresh, seed, quiet, cores, trials,
                             iterations, rssd_zero_penalty, objective_weights,
                             nevergrad_algo, intercept, intercept_sign, lambda_control,
                             outputs)

    # Perform assertions on the output
    assert isinstance(result, dict)
    assert 'OutputModels' in result
    assert 'OutputCollect' in result

    assert isinstance(result['OutputModels'], dict)
    assert isinstance(result['OutputCollect'], pd.DataFrame)

    assert 'metadata' in result['OutputModels']
    assert 'hyper_fixed' in result['OutputModels']['metadata']
    assert 'bootstrap' in result['OutputModels']['metadata']
    assert 'refresh' in result['OutputModels']['metadata']
    assert 'train_timestamp' in result['OutputModels']['metadata']
    assert 'cores' in result['OutputModels']['metadata']
    assert 'iterations' in result['OutputModels']['metadata']
    assert 'trials' in result['OutputModels']['metadata']
    assert 'intercept' in result['OutputModels']['metadata']
    assert 'intercept_sign' in result['OutputModels']['metadata']
    assert 'nevergrad_algo' in result['OutputModels']['metadata']
    assert 'ts_validation' in result['OutputModels']['metadata']
    assert 'add_penalty_factor' in result['OutputModels']['metadata']
    assert 'hyper_updated' in result['OutputModels']['metadata']

    assert isinstance(result['OutputModels']['metadata']['hyper_fixed'], dict)
    assert isinstance(result['OutputModels']['metadata']['bootstrap'], pd.DataFrame)
    assert isinstance(result['OutputModels']['metadata']['refresh'], bool)
    assert isinstance(result['OutputModels']['metadata']['train_timestamp'], float)
    assert isinstance(result['OutputModels']['metadata']['cores'], int)
    assert isinstance(result['OutputModels']['metadata']['iterations'], int)
    assert isinstance(result['OutputModels']['metadata']['trials'], int)
    assert isinstance(result['OutputModels']['metadata']['intercept'], bool)
    assert isinstance(result['OutputModels']['metadata']['intercept_sign'], str)
    assert isinstance(result['OutputModels']['metadata']['nevergrad_algo'], str)
    assert isinstance(result['OutputModels']['metadata']['ts_validation'], bool)
    assert isinstance(result['OutputModels']['metadata']['add_penalty_factor'], bool)
    assert isinstance(result['OutputModels']['metadata']['hyper_updated'], list)

    assert 'convergence' in result
    assert 'ts_validation_plot' in result

    assert isinstance(result['convergence'], pd.DataFrame)
    assert isinstance(result['ts_validation_plot'], pd.DataFrame)

    assert 'hyper_updated' in result
    assert 'seed' in result

    assert isinstance(result['hyper_updated'], list)
    assert isinstance(result['seed'], int)

    # Add more assertions as needed

@pytest.mark.xfail(raises=NotImplementedError)
def test_robyn_train():
    # Create dummy data
    InputCollect = {...}  # Replace with your actual input data
    hyper_collect = {...}  # Replace with your actual hyperparameter data
    cores = 4
    iterations = 100
    trials = 5
    intercept_sign = True
    intercept = 0.5
    nevergrad_algo = "NGT"
    dt_hyper_fixed = {...}  # Replace with your actual fixed hyperparameter data
    ts_validation = True
    add_penalty_factor = False
    objective_weights = {...}  # Replace with your actual objective weights
    rssd_zero_penalty = True
    refresh = False
    seed = 123
    quiet = False

    # Call the function
    result = robyn_train(InputCollect, hyper_collect, cores, iterations, trials,
                         intercept_sign, intercept, nevergrad_algo, dt_hyper_fixed,
                         ts_validation, add_penalty_factor, objective_weights,
                         rssd_zero_penalty, refresh, seed, quiet)

    # Perform assertions on the output
    assert isinstance(result, dict)
    assert 'trials' in result
    assert 'metadata' in result

    assert isinstance(result['trials'], list)
    assert len(result['trials']) == trials

    for trial in result['trials']:
        assert isinstance(trial, dict)
        assert 'trial' in trial
        assert 'name' in trial

        assert isinstance(trial['trial'], int)
        assert isinstance(trial['name'], str)

    assert 'solID' in result['trials'][0]['resultCollect']['xDecompVec']
    assert 'solID' in result['trials'][0]['resultCollect']['xDecompAgg']
    assert 'solID' in result['trials'][0]['resultCollect']['decompSpendDist']

    # Add more assertions as needed

@pytest.mark.xfail(raises=NotImplementedError)
def test_robyn_mmm():
    # Create dummy data
    InputCollect = {
        'dt_mod': pd.DataFrame({'ds': [1, 2, 3, 4, 5], 'dep_var': [0.1, 0.2, 0.3, 0.4, 0.5]}),
        'xDecompAggPrev': pd.DataFrame({'x': [0.1, 0.2, 0.3, 0.4, 0.5]}),
        'rollingWindowStartWhich': 1,
        'rollingWindowEndWhich': 5,
        'refreshAddedStart': 1,
        'dt_modRollWind': pd.DataFrame({'ds': [1, 2, 3, 4, 5]}),
        'refresh_steps': 1,
        'rollingWindowLength': 5,
        'paid_media_spends': ['col1', 'col2'],
        'organic_vars': ['col3', 'col4'],
        'context_vars': ['col5', 'col6'],
        'prophet_vars': ['col7', 'col8'],
        'adstock': 0.5,
        'context_signs': [1, -1],
        'paid_media_signs': [1, -1],
        'prophet_signs': [1, -1],
        'organic_signs': [1, -1],
        'calibration_input': None,
        'nevergrad_algo': 'NGalgo',
        'hyper_fixed': {'hyper_fixed': False},
        'iterations': 10,
        'cores': 4,
        'intercept_sign': 1,
        'intercept': True,
        'ts_validation': True,
        'add_penalty_factor': False,
        'objective_weights': None,
        'dt_hyper_fixed': None,
        'rssd_zero_penalty': True,
        'refresh': False,
        'trial': 1,
        'seed': 123,
        'quiet': False
    }

    # Call the function
    result = robyn_mmm(**InputCollect)

    # Perform assertions on the output
    assert isinstance(result, list)
    assert len(result) == InputCollect['iterations']

    # Add more assertions as needed

@pytest.mark.xfail(raises=NotImplementedError)
def test_model_decomp():
    # Create dummy data
    coefs = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    dt_modSaturated = pd.DataFrame({'dep_var': [1, 2, 3, 4, 5],
                                    'col1': [0.1, 0.2, 0.3, 0.4, 0.5],
                                    'col2': [0.5, 0.4, 0.3, 0.2, 0.1]})
    dt_saturatedImmediate = pd.DataFrame({'season': [1, 2, 3, 4, 5],
                                          'competitor_sales_B': [0.1, 0.2, 0.3, 0.4, 0.5]})
    dt_saturatedCarryover = pd.DataFrame({'season': [1, 2, 3, 4, 5],
                                          'competitor_sales_B': [0.5, 0.4, 0.3, 0.2, 0.1]})
    dt_modRollWind = pd.DataFrame({'ds': [1, 2, 3, 4, 5]})
    refreshAddedStart = 1

    # Call the function
    result = model.model_decomp(coefs, y_pred, dt_modSaturated, dt_saturatedImmediate,
                          dt_saturatedCarryover, dt_modRollWind, refreshAddedStart)

    # Perform assertions on the output
    assert isinstance(result, dict)
    assert 'xDecompVec' in result
    assert 'xDecompVec.scaled' in result
    assert 'xDecompAgg' in result
    assert 'coefsOutCat' in result
    assert 'mediaDecompImmediate' in result
    assert 'mediaDecompCarryover' in result

    assert isinstance(result['xDecompVec'], pd.DataFrame)
    assert isinstance(result['xDecompVec.scaled'], pd.DataFrame)
    assert isinstance(result['xDecompAgg'], pd.DataFrame)
    assert isinstance(result['coefsOutCat'], pd.DataFrame)
    assert isinstance(result['mediaDecompImmediate'], pd.DataFrame)
    assert isinstance(result['mediaDecompCarryover'], pd.DataFrame)

    assert result['xDecompVec'].shape[0] == dt_modSaturated.shape[0]
    assert result['xDecompVec.scaled'].shape[0] == dt_modSaturated.shape[0]
    assert result['xDecompAgg'].shape[0] == len(coefs) - 1
    assert result['coefsOutCat'].shape[0] == len(coefs)
    assert result['mediaDecompImmediate'].shape[0] == dt_saturatedImmediate.shape[0]
    assert result['mediaDecompCarryover'].shape[0] == dt_saturatedCarryover.shape[0]

    assert 'xDecompAgg' in result['xDecompAgg']
    assert 'xDecompPerc' in result['xDecompAgg']
    assert 'xDecompMeanNon0' in result['xDecompAgg']
    assert 'xDecompMeanNon0Perc' in result['xDecompAgg']
    assert 'xDecompAggRF' in result['xDecompAgg']
    assert 'xDecompPercRF' in result['xDecompAgg']
    assert 'xDecompMeanNon0RF' in result['xDecompAgg']
    assert 'xDecompMeanNon0PercRF' in result['xDecompAgg']
    assert 'pos' in result['xDecompAgg']

    assert 'rn' in result['coefsOutCat']
    assert 'coefs' in result['coefsOutCat']

    assert 'ds' in result['mediaDecompImmediate']
    assert 'y' in result['mediaDecompImmediate']

    assert 'ds' in result['mediaDecompCarryover']
    assert 'y' in result['mediaDecompCarryover']

    # Add more assertions as needed

@pytest.mark.xfail(raises=NotImplementedError)
def test_model_refit():
    # Create dummy data
    x_train = pd.DataFrame({'col1': [1, 2, 3, 4, 5],
                            'col2': [0.1, 0.2, 0.3, 0.4, 0.5]})
    y_train = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    x_val = pd.DataFrame({'col1': [6, 7, 8, 9, 10],
                          'col2': [0.6, 0.7, 0.8, 0.9, 1.0]})
    y_val = np.array([0.6, 0.7, 0.8, 0.9, 1.0])
    x_test = pd.DataFrame({'col1': [11, 12, 13, 14, 15],
                           'col2': [1.1, 1.2, 1.3, 1.4, 1.5]})
    y_test = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
    lambda_ = 0.1
    lower_limits = None
    upper_limits = None
    intercept = True
    intercept_sign = "non_negative"

    # Call the function
    result = model.model_refit(x_train, y_train, x_val, y_val, x_test, y_test, lambda_, lower_limits, upper_limits, intercept, intercept_sign)

    # Perform assertions on the output
    assert isinstance(result, dict)
    assert 'rsq_train' in result
    assert 'rsq_val' in result
    assert 'rsq_test' in result
    assert 'nrmse_train' in result
    assert 'nrmse_val' in result
    assert 'nrmse_test' in result
    assert 'coefs' in result
    assert 'y_train_pred' in result
    assert 'y_val_pred' in result
    assert 'y_test_pred' in result
    assert 'y_pred' in result
    assert 'mod' in result
    assert 'df_int' in result

    assert isinstance(result['rsq_train'], float)
    assert isinstance(result['rsq_val'], float) or result['rsq_val'] is None
    assert isinstance(result['rsq_test'], float) or result['rsq_test'] is None
    assert isinstance(result['nrmse_train'], float)
    assert isinstance(result['nrmse_val'], float) or result['nrmse_val'] is None
    assert isinstance(result['nrmse_test'], float) or result['nrmse_test'] is None
    assert isinstance(result['coefs'], pd.DataFrame)
    assert isinstance(result['y_train_pred'], np.ndarray)
    assert isinstance(result['y_val_pred'], np.ndarray) or result['y_val_pred'] is None
    assert isinstance(result['y_test_pred'], np.ndarray) or result['y_test_pred'] is None
    assert isinstance(result['y_pred'], np.ndarray)
    assert isinstance(result['mod'], Ridge)
    assert isinstance(result['df_int'], bool)

    assert result['coefs'].shape[0] == x_train.shape[1] + 1  # +1 for intercept
    assert result['y_train_pred'].shape[0] == y_train.shape[0]
    assert result['y_val_pred'] is None or result['y_val_pred'].shape[0] == y_val.shape[0]
    assert result['y_test_pred'] is None or result['y_test_pred'].shape[0] == y_test.shape[0]
    assert result['y_pred'].shape[0] == y_train.shape[0] + (y_val.shape[0] if y_val is not None else 0) + (y_test.shape[0] if y_test is not None else 0)

# Run the tests
pytest.main(["-v", "--tb=line", "~/Documents/GitHub/Robyn/python/python/tests/test_model.py"])