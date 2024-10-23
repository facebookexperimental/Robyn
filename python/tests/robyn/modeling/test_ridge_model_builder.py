import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
from robyn.modeling.ridge_model_builder import RidgeModelBuilder
from robyn.data.entities.hyperparameters import Hyperparameters
from robyn.modeling.entities.modelrun_trials_config import TrialsConfig
from robyn.modeling.entities.enums import NevergradAlgorithm
from robyn.modeling.entities.modeloutputs import ModelOutputs, Trial


class TestRidgeModelBuilder(unittest.TestCase):

    @patch("robyn.modeling.fridge_model_builder.Convergence")
    @patch("robyn.modeling.fridge_model_builder.RidgeModelBuilder._hyper_collector")
    @patch("robyn.modeling.fridge_model_builder.RidgeModelBuilder._model_train")
    @patch("robyn.modeling.fridge_model_builder.RidgeModelBuilder._select_best_model")
    def test_build_models(
        self,
        mock_select_best_model,
        mock_model_train,
        mock_hyper_collector,
        mock_convergence,
    ):
        # Step 1: Initialize the RidgeModelBuilder with mock dependencies and inputs
        mock_mmm_data = MagicMock()
        mock_holiday_data = MagicMock()
        mock_calibration_input = MagicMock()
        mock_hyperparameters = MagicMock(spec=Hyperparameters)
        mock_featurized_mmm_data = MagicMock()
        builder = RidgeModelBuilder(
            mock_mmm_data,
            mock_holiday_data,
            mock_calibration_input,
            mock_hyperparameters,
            mock_featurized_mmm_data,
        )

        # Step 2: Mock the Convergence class's calculate_convergence method to return a successful convergence message
        mock_convergence_instance = mock_convergence.return_value
        mock_convergence_instance.calculate_convergence.return_value = {
            "conv_msg": ["Convergence successful"]
        }

        # Step 3: Mock the _hyper_collector method to simulate hyperparameter collection with default parameters
        mock_hyper_collector.return_value = {
            "hyper_list_all": [],
            "hyper_bound_list_updated": [],
            "hyper_bound_list_fixed": [],
            "dt_hyper_fixed_mod": pd.DataFrame(),
            "all_fixed": False,
        }

        # Step 4: Mock the _model_train method to return a list of mock Trial objects
        mock_trial = MagicMock(spec=Trial)
        mock_model_train.return_value = [mock_trial]

        # Step 5: Mock the _select_best_model method to return a mock best model ID
        mock_select_best_model.return_value = "best_model_id"

        # Step 6: Call the build_models method with the specified input parameters
        trials_config = TrialsConfig(trials=3, iterations=10)
        model_outputs = builder.build_models(trials_config)

        # Step 7: Assert that model_outputs.trials matches the expected list of Trial objects
        self.assertEqual(model_outputs.trials, [mock_trial])

        # Step 8: Assert that model_outputs.convergence.conv_msg contains the expected convergence message
        self.assertIn("Convergence successful", model_outputs.convergence["conv_msg"])

        # Step 9: Assert that model_outputs.select_id matches the expected best model ID
        self.assertEqual(model_outputs.select_id, "best_model_id")

    def test__select_best_model(self):
        # Step 1: Provide a list of mock Trial objects with distinct nrmse and decomp_rssd values
        mock_trial_1 = MagicMock(spec=Trial)
        mock_trial_1.nrmse = 0.1
        mock_trial_1.decomp_rssd = 0.2
        mock_trial_1.result_hyp_param = {"solID": pd.Series(["model_1"])}

        mock_trial_2 = MagicMock(spec=Trial)
        mock_trial_2.nrmse = 0.2
        mock_trial_2.decomp_rssd = 0.1
        mock_trial_2.result_hyp_param = {"solID": pd.Series(["model_2"])}

        output_models = [mock_trial_1, mock_trial_2]

        # Step 2: Call the _select_best_model method with the provided input models
        builder = RidgeModelBuilder(None, None, None, None, None)
        best_model_id = builder._select_best_model(output_models)

        # Step 3: Assert that the returned solID matches the expected value based on the lowest combined score
        self.assertEqual(best_model_id, "model_1")

        # Step 4: Repeat the test with models having the same nrmse and decomp_rssd values
        mock_trial_3 = MagicMock(spec=Trial)
        mock_trial_3.nrmse = 0.1
        mock_trial_3.decomp_rssd = 0.1
        mock_trial_3.result_hyp_param = {"solID": pd.Series(["model_3"])}

        output_models = [mock_trial_1, mock_trial_3]

        # Step 5: Assert that the returned solID defaults to the first model in case of a tie
        best_model_id = builder._select_best_model(output_models)
        self.assertEqual(best_model_id, "model_1")

        # Step 6: Test with models having inversely proportional nrmse and decomp_rssd values
        mock_trial_4 = MagicMock(spec=Trial)
        mock_trial_4.nrmse = 0.05
        mock_trial_4.decomp_rssd = 0.25
        mock_trial_4.result_hyp_param = {"solID": pd.Series(["model_4"])}

        output_models = [mock_trial_4, mock_trial_2]

        # Step 7: Assert that the returned solID aligns with the expected best model based on combined score
        best_model_id = builder._select_best_model(output_models)
        self.assertEqual(best_model_id, "model_2")

    @patch(
        "robyn.modeling.fridge_model_builder.RidgeModelBuilder._run_nevergrad_optimization"
    )
    def test__model_train(self, mock_run_nevergrad_optimization):
        # Step 1: Mock the _run_nevergrad_optimization to simulate successful trial execution
        mock_trial = MagicMock(spec=Trial)
        mock_run_nevergrad_optimization.return_value = mock_trial

        # Step 2: Call the _model_train method with valid parameters where all trials should succeed
        builder = RidgeModelBuilder(None, None, None, None, None)
        trials_config = TrialsConfig(trials=2, iterations=5)
        trials = builder._model_train(
            {},
            trials_config,
            "non_negative",
            True,
            NevergradAlgorithm.TWO_POINTS_DE,
            None,
            False,
            False,
            None,
            True,
            123,
            2,
        )

        # Step 3: Assert that the length of trials matches the expected trial count
        self.assertEqual(len(trials), 2)

        # Step 4: Assert that each trial result indicates success
        self.assertTrue(all(isinstance(trial, Trial) for trial in trials))

        # Step 5: Test with zero trials to ensure it handles gracefully
        trials_config = TrialsConfig(trials=0, iterations=5)
        trials = builder._model_train(
            {},
            trials_config,
            "non_negative",
            True,
            NevergradAlgorithm.TWO_POINTS_DE,
            None,
            False,
            False,
            None,
            True,
            123,
            2,
        )

        # Step 6: Assert that the trials list is empty when no trials are configured
        self.assertEqual(len(trials), 0)

        # Step 7: Test with add_penalty_factor enabled to verify behavior change
        trials = builder._model_train(
            {},
            trials_config,
            "non_negative",
            True,
            NevergradAlgorithm.TWO_POINTS_DE,
            None,
            False,
            True,
            None,
            True,
            123,
            2,
        )

        # Step 8: Assert that trial results include penalty factor success indications
        self.assertTrue(all(isinstance(trial, Trial) for trial in trials))

    @patch("robyn.modeling.fridge_model_builder.ng.optimizers.registry")
    def test__run_nevergrad_optimization(self, mock_optimizer_registry):
        # Step 1: Mock the ng.optimizers.registry to simulate optimizer instance creation
        mock_optimizer_instance = MagicMock()
        mock_optimizer_registry.return_value = mock_optimizer_instance

        # Step 2: Call _run_nevergrad_optimization with basic valid inputs
        builder = RidgeModelBuilder(None, None, None, None, None)
        hyper_collect = {
            "hyper_bound_list_updated": {"param1": (0, 1)},
            "hyper_bound_list_fixed": {},
        }
        trial = builder._run_nevergrad_optimization(
            hyper_collect,
            10,
            2,
            NevergradAlgorithm.TWO_POINTS_DE,
            True,
            "non_negative",
            False,
            False,
            None,
            None,
            True,
            1,
            123,
            3,
        )

        # Step 3: Assert that the optimizer is called with correct parameters
        mock_optimizer_registry.assert_called_once_with(
            NevergradAlgorithm.TWO_POINTS_DE.value,
            mock_optimizer_instance,
            budget=10,
            num_workers=2,
        )

        # Step 4: Assert that the best trial parameters match expected values
        self.assertTrue(isinstance(trial, Trial))

        # Step 5: Test with no cores specified, ensuring single-core execution
        trial = builder._run_nevergrad_optimization(
            hyper_collect,
            10,
            1,
            NevergradAlgorithm.TWO_POINTS_DE,
            True,
            "non_negative",
            False,
            False,
            None,
            None,
            True,
            1,
            123,
            3,
        )

        # Step 6: Assert that the optimizer operates correctly with a single worker
        mock_optimizer_registry.assert_called_with(
            NevergradAlgorithm.TWO_POINTS_DE.value,
            mock_optimizer_instance,
            budget=10,
            num_workers=1,
        )

        # Step 7: Test with fixed hyperparameters and ensure optimization respects fixed values
        hyper_collect["hyper_bound_list_fixed"]["param1"] = 0.5
        trial = builder._run_nevergrad_optimization(
            hyper_collect,
            10,
            2,
            NevergradAlgorithm.TWO_POINTS_DE,
            True,
            "non_negative",
            False,
            False,
            None,
            None,
            True,
            1,
            123,
            3,
        )

        # Step 8: Assert that optimization outputs reflect fixed parameters
        self.assertTrue(isinstance(trial, Trial))

    def test__prepare_data(self):
        builder = RidgeModelBuilder(None, None, None, None, None)

        # Step 1: Call _prepare_data with empty parameters and empty dataframes
        X, y = builder._prepare_data({})

        # Step 2: Assert that the output X is a pandas DataFrame with zero length
        self.assertEqual(len(X), 0)

        # Step 3: Assert that the output y is a pandas Series with zero length
        self.assertEqual(len(y), 0)

        # Step 4: Test with parameters that do not match any media spends
        builder.mmm_data = MagicMock()
        builder.mmm_data.mmmdata_spec.paid_media_spends = []
        X, y = builder._prepare_data({"non_existent_media": 0.5})

        # Step 5: Assert that X and y remain unchanged from their original versions
        self.assertEqual(len(X), 0)
        self.assertEqual(len(y), 0)

        # Step 6: Test with valid parameters affecting media spend columns
        mock_media_spend = pd.Series([1, 2, 3])
        builder.featurized_mmm_data = MagicMock()
        builder.featurized_mmm_data.dt_mod = pd.DataFrame(
            {"media_spend": mock_media_spend}
        )
        builder.mmm_data.mmmdata_spec.paid_media_spends = ["media_spend"]

        # Mock transformations to verify correct application
        with patch.object(
            builder, "_geometric_adstock", return_value=mock_media_spend * 2
        ) as mock_adstock:
            X, y = builder._prepare_data({"media_spend_thetas": 0.5})

        # Step 8: Assert that transformed media spend matches expected series
        self.assertTrue((X["media_spend"] == mock_media_spend * 2).all())

        # Step 9: Test for handling NaN and infinite values
        mock_media_spend = pd.Series([1, np.nan, np.inf])
        builder.featurized_mmm_data.dt_mod = pd.DataFrame(
            {"media_spend": mock_media_spend}
        )
        X, y = builder._prepare_data({"media_spend_thetas": 0.5})

        # Step 10: Assert that output data does not contain NaN or infinite values
        self.assertFalse(X.isnull().any().any())
        self.assertFalse(np.isinf(X).any().any())

    def test__geometric_adstock(self):
        builder = RidgeModelBuilder(None, None, None, None, None)

        # Step 1: Provide a small array and theta=0.5 for basic functionality testing
        input_series = pd.Series([1, 2, 3])
        theta = 0.5

        # Step 2: Call _geometric_adstock with the provided inputs
        result = builder._geometric_adstock(input_series, theta)

        # Step 3: Assert that the output matches the expected transformed series
        expected_result = pd.Series([1, 2.5, 4.25])
        self.assertTrue(result.equals(expected_result))

        # Step 4: Test with an empty series for edge case handling
        input_series = pd.Series([])
        result = builder._geometric_adstock(input_series, theta)

        # Step 5: Assert that the output remains an empty series
        self.assertTrue(result.equals(pd.Series([])))

        # Step 6: Test with theta=0 to ensure no decay effect
        input_series = pd.Series([1, 2, 3])
        result = builder._geometric_adstock(input_series, 0)

        # Step 7: Assert that the output matches the input series
        self.assertTrue(result.equals(input_series))

        # Step 8: Test with theta=1 for full decay effect
        result = builder._geometric_adstock(input_series, 1)

        # Step 9: Assert that the output series reflects expected full decay transformation
        expected_result = pd.Series([1, 3, 6])
        self.assertTrue(result.equals(expected_result))

        # Step 10: Test with negative numbers in the series
        input_series = pd.Series([-1, -2, -3])
        result = builder._geometric_adstock(input_series, theta)

        # Step 11: Assert that the output series matches expected transformation for negative values
        expected_result = pd.Series([-1, -2.5, -4.25])
        self.assertTrue(result.equals(expected_result))

        # Step 12: Test with theta greater than 1
        result = builder._geometric_adstock(input_series, 1.5)

        # Step 13: Assert that the output series reflects transformation with increased decay
        expected_result = pd.Series([-1, -3.5, -8.75])
        self.assertTrue(result.equals(expected_result))

    def test__hill_transformation(self):
        builder = RidgeModelBuilder(None, None, None, None, None)

        # Step 1: Provide a series of normal values and parameters for basic testing
        input_series = pd.Series([0.1, 0.5, 0.9])
        alpha = 2
        gamma = 1

        # Step 2: Call _hill_transformation with the given inputs
        result = builder._hill_transformation(input_series, alpha, gamma)

        # Step 3: Assert that the output matches the expected transformed series
        expected_result = pd.Series([0.00990099, 0.2, 0.45])
        pd.testing.assert_series_equal(result, expected_result, check_exact=False)

        # Step 4: Test with alpha as zero
        result = builder._hill_transformation(input_series, 0, gamma)

        # Step 5: Assert that the output series reflects uniform transformation due to zero alpha
        expected_result = pd.Series([0.5, 0.5, 0.5])
        pd.testing.assert_series_equal(result, expected_result, check_exact=False)

        # Step 6: Test with gamma as zero
        result = builder._hill_transformation(input_series, alpha, 0)

        # Step 7: Assert that the output series reflects transformation with gamma effect
        expected_result = pd.Series([1, 1, 1])
        pd.testing.assert_series_equal(result, expected_result, check_exact=False)

        # Step 8: Test with x having all identical values
        input_series = pd.Series([0.5, 0.5, 0.5])
        result = builder._hill_transformation(input_series, alpha, gamma)

        # Step 9: Assert that the output series contains NaN due to division by zero
        self.assertTrue(result.isna().all())

        # Step 10: Test with an empty series
        input_series = pd.Series([])
        result = builder._hill_transformation(input_series, alpha, gamma)

        # Step 11: Assert that the output remains an empty series
        self.assertTrue(result.empty)

        # Step 12: Test with large alpha and gamma
        input_series = pd.Series([0.1, 0.5, 0.9])
        result = builder._hill_transformation(input_series, 100, 100)

        # Step 13: Assert that the output series values are transformed close to zero
        self.assertTrue((result < 1e-10).all())

    def test__calculate_rssd(self):
        builder = RidgeModelBuilder(None, None, None, None, None)

        # Step 1: Provide coefficients and RSSD without zero penalty
        coefs = np.array([1, 2, 3])
        rssd_zero_penalty = False
        rssd = builder._calculate_rssd(coefs, rssd_zero_penalty)

        # Step 2: Assert that the calculated RSSD matches the expected value
        expected_rssd = np.sqrt(np.sum(coefs**2))
        self.assertEqual(rssd, expected_rssd)

        # Step 4: Test with zero penalty enabled
        rssd_zero_penalty = True
        rssd = builder._calculate_rssd(coefs, rssd_zero_penalty)

        # Step 5: Assert that the RSSD reflects zero coefficient penalty
        expected_rssd *= 1
        self.assertEqual(rssd, expected_rssd)

        # Step 6: Test with all zero coefficients
        coefs = np.array([0, 0, 0])
        rssd = builder._calculate_rssd(coefs, rssd_zero_penalty)

        # Step 7: Assert that the RSSD is zero
        self.assertEqual(rssd, 0)

        # Step 8: Test with mixed coefficients
        coefs = np.array([1, 0, -3])
        rssd = builder._calculate_rssd(coefs, rssd_zero_penalty)

        # Step 9: Assert that the RSSD matches expected value for mixed coefficients
        expected_rssd = np.sqrt(np.sum(coefs**2))
        self.assertEqual(rssd, expected_rssd)

        # Step 10: Test with large coefficients
        coefs = np.array([10, 20, 30])
        rssd = builder._calculate_rssd(coefs, rssd_zero_penalty)

        # Step 11: Assert that the RSSD reflects the magnitude of coefficients
        expected_rssd = np.sqrt(np.sum(coefs**2))
        self.assertEqual(rssd, expected_rssd)

        # Step 12: Test with a single coefficient
        coefs = np.array([10])
        rssd = builder._calculate_rssd(coefs, rssd_zero_penalty)

        # Step 13: Assert that the RSSD equals the absolute value of the coefficient
        self.assertEqual(rssd, np.abs(coefs[0]))

        # Step 14: Test with negative coefficients
        coefs = np.array([-10, -20, -30])
        rssd = builder._calculate_rssd(coefs, rssd_zero_penalty)

        # Step 15: Assert that the RSSD matches expected value for negative coefficients
        expected_rssd = np.sqrt(np.sum(coefs**2))
        self.assertEqual(rssd, expected_rssd)

    def test__calculate_mape(self):
        builder = RidgeModelBuilder(None, None, None, None, None)

        # Step 1: Mock dependencies to simulate valid calibration data
        builder.calibration_input = {
            "calibration_point": {
                "liftStartDate": "2020-01-01",
                "liftEndDate": "2020-01-31",
                "liftMedia": "media_spend",
            }
        }
        builder.mmm_data = MagicMock()
        builder.mmm_data.data = pd.DataFrame(
            {
                "date": pd.date_range(start="2020-01-01", periods=31),
                "media_spend": np.random.rand(31),
            }
        )
        builder.featurized_mmm_data = MagicMock()
        builder.featurized_mmm_data.rollingWindowStartWhich = 0
        builder.featurized_mmm_data.rollingWindowEndWhich = 30
        builder.mmm_data.mmmdata_spec = MagicMock()
        builder.mmm_data.mmmdata_spec.date_var = "date"
        builder.mmm_data.mmmdata_spec.dep_var = "media_spend"
        builder._prepare_features = MagicMock(return_value=np.random.rand(31, 3))

        # Step 2: Call _calculate_mape with a Ridge model instance
        mock_model = MagicMock()
        mock_model.predict = MagicMock(return_value=np.random.rand(31))
        mape = builder._calculate_mape(mock_model)

        # Step 3: Assert that the calculated MAPE matches the expected value
        self.assertTrue(isinstance(mape, float))

        # Step 4: Test with no calibration input
        builder.calibration_input = None
        mape = builder._calculate_mape(mock_model)

        # Step 5: Assert that the MAPE is zero when no calibration data is present
        self.assertEqual(mape, 0.0)

        # Step 6: Test with empty calibration data
        builder.calibration_input = {}
        mape = builder._calculate_mape(mock_model)

        # Step 7: Assert that the MAPE remains zero for empty calibration data
        self.assertEqual(mape, 0.0)

        # Step 8: Test with multiple calibration data points
        builder.calibration_input = {
            "calibration_point_1": {
                "liftStartDate": "2020-01-01",
                "liftEndDate": "2020-01-15",
                "liftMedia": "media_spend",
            },
            "calibration_point_2": {
                "liftStartDate": "2020-01-16",
                "liftEndDate": "2020-01-31",
                "liftMedia": "media_spend",
            },
        }
        mape = builder._calculate_mape(mock_model)

        # Step 9: Assert that the MAPE reflects the mean value across all calibration points
        self.assertTrue(isinstance(mape, float))

    @patch("robyn.modeling.fridge_model_builder.RidgeModelBuilder._prepare_data")
    def test__evaluate_model(self, mock_prepare_data):
        builder = RidgeModelBuilder(None, None, None, None, None)

        # Step 1: Mock dependencies to simulate valid input data and outputs
        X = pd.DataFrame(np.random.rand(100, 5))
        y = pd.Series(np.random.rand(100))
        mock_prepare_data.return_value = (X, y)

        # Step 2: Call _evaluate_model with specified parameters
        params = {"lambda": 0.1}
        loss, nrmse, decomp_rssd, mape, *_ = builder._evaluate_model(
            params, False, False, False, None
        )

        # Step 3: Assert that loss and evaluation metrics fall within expected ranges
        self.assertTrue(0 <= loss <= 1)
        self.assertTrue(0 <= nrmse <= 1)
        self.assertTrue(0 <= decomp_rssd <= 1)
        self.assertTrue(0 <= mape <= 100 or mape is None)

        # Step 4: Test with time-series validation enabled
        loss, nrmse, decomp_rssd, mape, *_ = builder._evaluate_model(
            params, True, False, False, None
        )

        # Step 5: Assert that evaluation metrics reflect validation results
        self.assertTrue(0 <= loss <= 1)

        # Step 6: Test with custom objective weights
        objective_weights = [0.5, 0.3, 0.2]
        loss, *_ = builder._evaluate_model(
            params, False, False, False, objective_weights
        )

        # Step 7: Assert that loss calculation respects custom weights
        self.assertTrue(0 <= loss <= 1)

        # Step 8: Test with missing optional parameters
        loss, *_ = builder._evaluate_model(params, False, False, False, None)

        # Step 9: Assert that defaults are correctly applied in evaluation
        self.assertTrue(0 <= loss <= 1)

    def test__hyper_collector(self):
        builder = RidgeModelBuilder(None, None, None, None, None)

        # Step 1: Call _hyper_collector with fixed hyperparameters provided
        hyperparameters = Hyperparameters(
            hyperparameters={"param1": [0, 1], "param2": 5}
        )
        result = builder._hyper_collector(hyperparameters, False, False, None, 2)

        # Step 2: Assert that hyper_list_all matches the expected collected hyperparameters
        self.assertIn("param1", result["hyper_bound_list_updated"])
        self.assertIn("param2", result["hyper_bound_list_fixed"])

        # Step 4: Test without fixed hyperparameters
        result = builder._hyper_collector(
            hyperparameters, False, False, pd.DataFrame(), 2
        )

        # Step 5: Assert that dt_hyper_fixed_mod is empty and all_fixed is False
        self.assertTrue(result["dt_hyper_fixed_mod"].empty)
        self.assertFalse(result["all_fixed"])

        # Step 6: Test with empty hyperparameters
        hyperparameters = Hyperparameters(hyperparameters={})
        result = builder._hyper_collector(hyperparameters, False, False, None, 2)

        # Step 7: Assert that all collections remain empty
        self.assertFalse(result["hyper_bound_list_updated"])
        self.assertFalse(result["hyper_bound_list_fixed"])

        # Step 8: Test with non-list parameter values
        hyperparameters = Hyperparameters(hyperparameters={"param1": 5})
        result = builder._hyper_collector(hyperparameters, False, False, None, 2)

        # Step 9: Assert that hyper_bound_list_fixed captures non-list values
        self.assertIn("param1", result["hyper_bound_list_fixed"])

    def test__model_refit(self):
        builder = RidgeModelBuilder(None, None, None, None, None)

        # Step 1: Call _model_refit with basic train data only
        x_train = np.random.rand(100, 5)
        y_train = np.random.rand(100)
        result = builder._model_refit(x_train, y_train)

        # Step 2: Assert that training metrics fall within expected ranges
        self.assertTrue(0 <= result.rsq_train <= 1)

        # Step 3: Test with train and validation data
        x_val = np.random.rand(20, 5)
        y_val = np.random.rand(20)
        result = builder._model_refit(x_train, y_train, x_val, y_val)

        # Step 4: Assert that validation metrics are correctly calculated
        self.assertTrue(0 <= result.rsq_val <= 1)

        # Step 5: Test with train, validation, and test data
        x_test = np.random.rand(20, 5)
        y_test = np.random.rand(20)
        result = builder._model_refit(x_train, y_train, x_val, y_val, x_test, y_test)

        # Step 6: Assert that all metrics are calculated for each dataset
        self.assertTrue(0 <= result.rsq_test <= 1)

        # Step 7: Test with intercept set to False
        result = builder._model_refit(x_train, y_train, intercept=False)

        # Step 8: Assert that model coefficients and predictions reflect intercept absence
        self.assertEqual(result.df_int, 0)

    def test__lambda_seq(self):
        builder = RidgeModelBuilder(None, None, None, None, None)

        # Step 1: Call _lambda_seq with default sequence length and lambda_min_ratio on small dataset
        x = np.random.rand(100, 5)
        y = np.random.rand(100)
        sequence = builder._lambda_seq(x, y)

        # Step 2: Assert that the length of the sequence matches the expected value
        self.assertEqual(len(sequence), 100)

        # Step 3: Assert that all values in the sequence are positive
        self.assertTrue(all(sequence > 0))

        # Step 4: Test with custom sequence length and lambda_min_ratio on larger dataset
        sequence = builder._lambda_seq(x, y, seq_len=50, lambda_min_ratio=0.001)

        # Step 5: Assert that the sequence length matches the specified custom length
        self.assertEqual(len(sequence), 50)

        # Step 6: Assert that the first value is greater than the last value in the sequence
        self.assertTrue(sequence[0] > sequence[-1])

        # Step 7: Test with x and y as zero arrays for edge handling
        x = np.zeros((100, 5))
        y = np.zeros(100)
        sequence = builder._lambda_seq(x, y)

        # Step 8: Assert that all sequence values are non-positive, reflecting edge case
        self.assertTrue(all(sequence <= 0))

        # Step 9: Test with high dimensional dataset and very small lambda_min_ratio
        x = np.random.rand(500, 50)
        y = np.random.rand(500)
        sequence = builder._lambda_seq(x, y, seq_len=200, lambda_min_ratio=0.00001)

        # Step 10: Assert that the sequence length matches the expected large length
        self.assertEqual(len(sequence), 200)

        # Step 11: Assert that all values in the sequence are positive
        self.assertTrue(all(sequence > 0))


if __name__ == "__main__":
    unittest.main()
