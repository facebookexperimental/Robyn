# pyre-strict

import unittest
from sklearn.linear_model import Ridge
from robyn.modeling.ridge_model_builder import RidgeModelBuilder
from robyn.data.entities.holidays_data import HolidaysData
from robyn.data.entities.mmmdata import MMMData
from robyn.modeling.entities.modeloutputs import Trial
from robyn.modeling.feature_engineering import FeaturizedMMMData
from robyn.modeling.convergence.convergence import Convergence
from robyn.modeling.entities.enums import NevergradAlgorithm
from robyn.modeling.entities.modelrun_trials_config import TrialsConfig
from unittest.mock import MagicMock
import numpy as np


class RidgeModelBuilderTest(unittest.TestCase):

    def setUp(self):
        self.ridge_model_builder = RidgeModelBuilder()

    def test_build_models_default_parameters(self):
        # Mock the _hyper_collector method
        self.ridge_model_builder._hyper_collector = MagicMock(
            return_value={
                "hyper_list_all": [],
                "all_fixed": [],
                "hyper_bound_list_updated": [],
                "hyper_bound_list_fixed": [],
            }
        )

        # Mock the _run_nevergrad_optimization method
        mock_trial = MagicMock(spec=Trial)
        self.ridge_model_builder._run_nevergrad_optimization = MagicMock(
            return_value=mock_trial
        )

        # Mock the calculate_convergence method
        mock_convergence_results = MagicMock()
        Convergence.calculate_convergence = MagicMock(
            return_value=mock_convergence_results
        )

        # Call the build_models method
        trials_config = TrialsConfig(trials=1, iterations=1)
        model_outputs = self.ridge_model_builder.build_models(
            trials_config=trials_config,
            dt_hyper_fixed=None,
            ts_validation=False,
            add_penalty_factor=False,
            seed=123,
            rssd_zero_penalty=True,
            objective_weights=None,
            nevergrad_algo=NevergradAlgorithm.TWO_POINTS_DE,
            intercept=True,
            intercept_sign="non_negative",
            cores=2,
        )

        # Assertions
        self.assertEqual(len(model_outputs.trials), 1)
        self.assertIs(model_outputs.trials[0], mock_trial)
        self.assertEqual(model_outputs.convergence, mock_convergence_results)
        self.assertEqual(model_outputs.hyper_updated, [])

    def test_build_models_with_ts_validation(self):
        # Mocking the methods
        with unittest.mock.patch.object(
            RidgeModelBuilder,
            "_hyper_collector",
            return_value={
                "hyper_list_all": [],
                "all_fixed": False,
                "hyper_bound_list_updated": [],
                "hyper_bound_list_fixed": [],
            },
        ) as mock_hyper_collector, unittest.mock.patch.object(
            RidgeModelBuilder,
            "_run_nevergrad_optimization",
            return_value=unittest.mock.Mock(),
        ) as mock_run_optimization, unittest.mock.patch.object(
            Convergence, "calculate_convergence", return_value="Convergence results"
        ) as mock_convergence:

            # Input parameters
            trials_config = TrialsConfig(trials=1, iterations=1)

            # Call build_models
            model_outputs = self.ridge_model_builder.build_models(
                trials_config=trials_config,
                dt_hyper_fixed=None,
                ts_validation=True,
                add_penalty_factor=False,
                seed=123,
                rssd_zero_penalty=True,
                objective_weights=None,
                nevergrad_algo=NevergradAlgorithm.TWO_POINTS_DE,
                intercept=True,
                intercept_sign="non_negative",
                cores=2,
            )

            # Assertions
            self.assertTrue(model_outputs.ts_validation)
            self.assertEqual(len(model_outputs.trials), 1)

    def test_build_models_with_penalty_factor(self):
        # Mock the _hyper_collector method
        ridge_model_builder = RidgeModelBuilder(
            ...
        )  # Presumed setup of RidgeModelBuilder instance
        ridge_model_builder._hyper_collector = MagicMock(
            return_value={
                "hyper_list_all": [],
                "all_fixed": False,
                "hyper_bound_list_updated": [],
                "hyper_bound_list_fixed": {},
            }
        )

        # Mock the _run_nevergrad_optimization method
        mock_trial = Trial(
            ...
        )  # Presumed setup of a mock Trial instance with expected attributes
        ridge_model_builder._run_nevergrad_optimization = MagicMock(
            return_value=mock_trial
        )

        # Mock Convergence.calculate_convergence method
        Convergence.calculate_convergence = MagicMock(
            return_value="Convergence results"
        )

        # Define the trials_config
        trials_config = TrialsConfig(trials=1, iterations=1)

        # Call build_models with the specified parameters
        model_outputs = ridge_model_builder.build_models(
            trials_config=trials_config,
            dt_hyper_fixed=None,
            ts_validation=False,
            add_penalty_factor=True,
            seed=123,
            rssd_zero_penalty=True,
            objective_weights=None,
            nevergrad_algo=NevergradAlgorithm.TWO_POINTS_DE,
            intercept=True,
            intercept_sign="non_negative",
            cores=2,
        )

        # Assert the expected outcomes
        self.assertTrue(model_outputs.add_penalty_factor)
        self.assertEqual(len(model_outputs.trials), 1)
        self.assertEqual(model_outputs.trials[0], mock_trial)

    def test_build_models_with_different_objective_weights(self):
        # Mock the _hyper_collector method
        self.ridge_model_builder._hyper_collector = (
            lambda hp, ts_val, add_penalty, dt_hyper, cores: {
                "hyper_list_all": [],
                "all_fixed": False,
                "hyper_bound_list_updated": [],
                "hyper_bound_list_fixed": {},
            }
        )

        # Create a mock Trial instance
        mock_trial = Trial(
            result_hyp_param=pd.DataFrame(),
            lift_calibration=pd.DataFrame(),
            decomp_spend_dist=pd.DataFrame(),
            nrmse=0.0,
            decomp_rssd=0.0,
            mape=0.0,
            x_decomp_agg=pd.DataFrame(),
            rsq_train=0.0,
            rsq_val=0.0,
            rsq_test=0.0,
            lambda_=0.0,
            lambda_hp=0.0,
            lambda_max=0.0,
            lambda_min_ratio=0.0,
            pos=False,
            elapsed=0.0,
            elapsed_accum=0.0,
            trial=1,
            iter_ng=1,
            iter_par=1,
            train_size=1.0,
            sol_id="1_1_1",
        )

        # Mock the _run_nevergrad_optimization method
        self.ridge_model_builder._run_nevergrad_optimization = (
            lambda *args, **kwargs: mock_trial
        )

        # Mock the calculate_convergence method
        Convergence.calculate_convergence = lambda self, trials: "Convergence results"

        # Run the build_models method
        model_outputs = self.ridge_model_builder.build_models(
            trials_config=TrialsConfig(trials=1, iterations=1),
            dt_hyper_fixed=None,
            ts_validation=False,
            add_penalty_factor=False,
            seed=123,
            rssd_zero_penalty=True,
            objective_weights=[0.3, 0.3, 0.4],
            nevergrad_algo=NevergradAlgorithm.TWO_POINTS_DE,
            intercept=True,
            intercept_sign="non_negative",
            cores=2,
        )

        # Assert the expected outputs
        self.assertEqual(model_outputs.objective_weights, [0.3, 0.3, 0.4])
        self.assertEqual(len(model_outputs.trials), 1)
        self.assertIsInstance(model_outputs.trials[0], Trial)

    def test_select_best_model_equal_metrics(self):
        # Create mock Trial objects with equal nrmse and decomp_rssd values
        mock_trials = [
            Trial(
                sol_id="model_1",
                nrmse=0.5,
                decomp_rssd=0.3,
                result_hyp_param={"solID": "model_1"},
            ),
            Trial(
                sol_id="model_2",
                nrmse=0.5,
                decomp_rssd=0.3,
                result_hyp_param={"solID": "model_2"},
            ),
            Trial(
                sol_id="model_3",
                nrmse=0.5,
                decomp_rssd=0.3,
                result_hyp_param={"solID": "model_3"},
            ),
        ]

        # Call the method under test
        best_model_sol_id = self.ridge_model_builder._select_best_model(mock_trials)

        # Assert that the returned solID is the first model's solID
        self.assertEqual(best_model_sol_id, "model_1")

    def test_select_best_model_distinct_metrics(self):
        # Create mock Trial objects with distinct metrics
        trial1 = Trial(
            nrmse=0.1, decomp_rssd=0.2, result_hyp_param={"solID": "model_1"}
        )
        trial2 = Trial(
            nrmse=0.3, decomp_rssd=0.3, result_hyp_param={"solID": "model_2"}
        )
        trial3 = Trial(
            nrmse=0.4, decomp_rssd=0.4, result_hyp_param={"solID": "model_3"}
        )

        # Create a list of these mock trials
        output_models = [trial1, trial2, trial3]

        # Call the method to test
        best_model_solID = self.ridge_model_builder._select_best_model(output_models)

        # Assert that the best model is the one with the lowest combined nrmse and decomp_rssd
        self.assertEqual(best_model_solID, "model_1")

    def test_select_best_model_negative_metrics(self):
        # Create mock output models with negative nrmse and decomp_rssd values
        trial_1 = Trial(
            nrmse=-0.5, decomp_rssd=-0.3, result_hyp_param={"solID": "model_1"}
        )
        trial_2 = Trial(
            nrmse=-0.4, decomp_rssd=-0.2, result_hyp_param={"solID": "model_2"}
        )
        trial_3 = Trial(
            nrmse=-0.6, decomp_rssd=-0.1, result_hyp_param={"solID": "model_3"}
        )
        output_models = [trial_1, trial_2, trial_3]

        # Use the _select_best_model method
        best_model_sol_id = self.ridge_model_builder._select_best_model(output_models)

        # Check if the best model is correctly identified
        self.assertEqual(best_model_sol_id, "model_2")

    def test_select_best_model_nan_metrics(self):
        # Mock data for trials with NaN values
        trials = [
            Trial(nrmse=np.nan, decomp_rssd=0.5, sol_id="model_1"),  # NaN nrmse
            Trial(nrmse=0.3, decomp_rssd=np.nan, sol_id="model_2"),  # NaN decomp_rssd
            Trial(nrmse=0.2, decomp_rssd=0.3, sol_id="model_3"),  # Valid model
            Trial(nrmse=np.nan, decomp_rssd=np.nan, sol_id="model_4"),  # Both NaN
        ]

        # Call the method under test
        best_model_solID = self.ridge_model_builder._select_best_model(trials)

        # Verify that the returned solID is for the model with valid metrics
        self.assertEqual(best_model_solID, "model_3")

    def test_select_best_model_single_model(self):
        # Create a mock Trial object with preset solID
        mock_trial = Trial(
            result_hyp_param=pd.DataFrame({"solID": ["model_1"]}),
            lift_calibration=pd.DataFrame(),
            decomp_spend_dist=pd.DataFrame(),
            nrmse=0.1,
            decomp_rssd=0.1,
            mape=0.1,
            x_decomp_agg=pd.DataFrame(),
            rsq_train=0.9,
            rsq_val=0.9,
            rsq_test=0.9,
            lambda_=1.0,
            lambda_hp=0.0,
            lambda_max=0.0,
            lambda_min_ratio=0.0001,
            pos=False,
            elapsed=0.0,
            elapsed_accum=0.0,
            trial=1,
            iter_ng=1,
            iter_par=1,
            train_size=1.0,
            sol_id="model_1",
        )
        # Single-element list of output_models
        output_models = [mock_trial]

        # Invoke the method with a single model
        best_model_id = self.ridge_model_builder._select_best_model(output_models)

        # Assert that the solID of the lone model is returned
        self.assertEqual(best_model_id, "model_1")

    def test_run_nevergrad_optimization_successful_optimization_with_valid_hyperparameters(
        self,
    ):
        # Mock the Nevergrad optimizer's ask and tell methods
        with unittest.mock.patch(
            "ng.optimizers.registry.ask"
        ) as mock_ask, unittest.mock.patch("ng.optimizers.registry.tell") as mock_tell:

            # Arrange: Set up the mock behavior
            mock_ask.return_value.kwargs = {"param1": 0.5, "param2": 1.0}
            mock_tell.return_value = None

            # Initialize input parameters
            hyper_collect = {
                "hyper_bound_list_updated": {
                    "param1": (0.1, 1.0),
                    "param2": (0.0, 2.0),
                },
                "hyper_list_all": {},
                "all_fixed": False,
            }
            iterations = 10
            cores = 2
            nevergrad_algo = NevergradAlgorithm.TWO_POINTS_DE
            intercept = True
            intercept_sign = "non_negative"
            ts_validation = False
            add_penalty_factor = False
            objective_weights = [0.5, 0.5]
            dt_hyper_fixed = None
            rssd_zero_penalty = True
            trial = 1
            seed = 123
            total_trials = 1

            # Act: Call the method under test
            result = self.ridge_model_builder._run_nevergrad_optimization(
                hyper_collect=hyper_collect,
                iterations=iterations,
                cores=cores,
                nevergrad_algo=nevergrad_algo,
                intercept=intercept,
                intercept_sign=intercept_sign,
                ts_validation=ts_validation,
                add_penalty_factor=add_penalty_factor,
                objective_weights=objective_weights,
                dt_hyper_fixed=dt_hyper_fixed,
                rssd_zero_penalty=rssd_zero_penalty,
                trial=trial,
                seed=seed,
                total_trials=total_trials,
            )

            # Assert: Check the expected outcomes
            self.assertEqual(result["loss"], 0.1)
            self.assertEqual(result.trial, 1)

    def test_run_nevergrad_optimization_optimization_with_zero_penalty_and_objective_weights(
        self,
    ):
        # Arrange
        hyper_collect = {
            "hyper_bound_list_updated": {"param1": (0, 1), "param2": (1, 2)}
        }
        iterations = 10
        cores = 1
        nevergrad_algo = NevergradAlgorithm.TWO_POINTS_DE
        intercept = True
        intercept_sign = "non_negative"
        ts_validation = False
        add_penalty_factor = False
        objective_weights = [0.1, 0.2, 0.7]
        dt_hyper_fixed = None
        rssd_zero_penalty = True
        trial = 2
        seed = 123
        total_trials = 5

        # Mock
        mock_ask = MagicMock(return_value=ng.p.Instrumentation(param1=0.3, param2=1.8))
        mock_tell = MagicMock()

        with patch("ng.optimizers.registry.ask", mock_ask), patch(
            "ng.optimizers.registry.tell", mock_tell
        ):
            # Act
            result = self.ridge_model_builder._run_nevergrad_optimization(
                hyper_collect=hyper_collect,
                iterations=iterations,
                cores=cores,
                nevergrad_algo=nevergrad_algo,
                intercept=intercept,
                intercept_sign=intercept_sign,
                ts_validation=ts_validation,
                add_penalty_factor=add_penalty_factor,
                objective_weights=objective_weights,
                dt_hyper_fixed=dt_hyper_fixed,
                rssd_zero_penalty=rssd_zero_penalty,
                trial=trial,
                seed=seed,
                total_trials=total_trials,
            )

        # Assert
        self.assertEqual(result["loss"], 0.05)
        self.assertEqual(result["trial"], 2)

    def test_run_nevergrad_optimization_with_different_algorithms(self):
        # Mock the ng.optimizers.registry.ask and tell methods
        mock_ask = self.patch(
            "ng.optimizers.registry.ask",
            return_value=ng.p.Instrumentation(param1=0.7, param2=1.5),
        )
        mock_tell = self.patch("ng.optimizers.registry.tell")

        # Define the input parameters for the test
        hyper_collect = {
            "hyper_bound_list_updated": {"param1": (0.0, 1.0), "param2": (1.0, 2.0)},
        }
        iterations = 10
        cores = 1
        nevergrad_algo = NevergradAlgorithm.ONE_PLUS_ONE
        intercept = True
        intercept_sign = "non_negative"
        ts_validation = False
        add_penalty_factor = False
        objective_weights = [0.5, 0.5]
        dt_hyper_fixed = None
        rssd_zero_penalty = True
        trial = 3
        seed = 123
        total_trials = 5

        # Invoke the _run_nevergrad_optimization method
        result = self.ridge_model_builder._run_nevergrad_optimization(
            hyper_collect=hyper_collect,
            iterations=iterations,
            cores=cores,
            nevergrad_algo=nevergrad_algo,
            intercept=intercept,
            intercept_sign=intercept_sign,
            ts_validation=ts_validation,
            add_penalty_factor=add_penalty_factor,
            objective_weights=objective_weights,
            dt_hyper_fixed=dt_hyper_fixed,
            rssd_zero_penalty=rssd_zero_penalty,
            trial=trial,
            seed=seed,
            total_trials=total_trials,
        )

        # Assert the expected outcomes
        self.assertEqual(result["loss"], 0.2)  # Ensure the loss value is as expected
        self.assertEqual(result["trial"], 3)  # Validate the trial tracking

    def test_calculate_decomp_spend_dist_standard_input(self):
        # Instantiate a Ridge model and mock its coefficients
        ridge_model = Ridge()
        ridge_model.coef_ = np.array([0.1, 0.2, 0.3])

        # Create DataFrame X and Series y with specified data
        X = pd.DataFrame(
            {"media1": [1, 2, 3], "media2": [4, 5, 6], "media3": [7, 8, 9]}
        )
        y = pd.Series([1, 2, 3])

        # Define the params dictionary
        params = {"rsq_val": 0.8, "rsq_test": 0.75, "lambda_": 0.01}

        # Call _calculate_decomp_spend_dist method
        result = self.ridge_model_builder._calculate_decomp_spend_dist(
            ridge_model, X, y, params
        )

        # Perform assertions
        self.assertAlmostEqual(result["rsq_train"][0], 0.95, places=2)
        self.assertEqual(result["rsq_val"][0], 0.8)
        self.assertEqual(result["rsq_test"][0], 0.75)
        self.assertEqual(result["lambda"][0], 0.01)

    def test_calculate_decomp_spend_dist_zero_coefficients(self) -> None:
        # Instantiate a Ridge model with zero coefficients
        model = Ridge()
        model.coef_ = np.zeros(3)  # Assuming there are 3 features

        # Create a DataFrame X with 3 columns and some sample data
        X = pd.DataFrame(
            {"feature1": [1, 2, 3], "feature2": [4, 5, 6], "feature3": [7, 8, 9]}
        )

        # Create a Series y with sample data
        y = pd.Series([10, 11, 12])

        # Define params dictionary with sample metrics
        params = {
            "rsq_val": 0.0,
            "rsq_test": 0.0,
            "nrmse_val": 0.0,
            "nrmse_test": 0.0,
            "nrmse": 0.0,
            "decomp_rssd": 0.0,
            "mape": 0.0,
            "lambda": 0.0,
            "solID": "",
            "trial": 0,
            "iter_ng": 0,
            "iter_par": 0,
        }

        # Call _calculate_decomp_spend_dist
        result = self.ridge_model_builder._calculate_decomp_spend_dist(
            model, X, y, params
        )

        # Assert that the sum of 'xDecompAgg' in the result DataFrame is zero
        self.assertEqual(result["xDecompAgg"].sum(), 0)

        # Assert that the sum of 'effect_share' in the result DataFrame is zero
        self.assertEqual(result["effect_share"].sum(), 0)

    def test_calculate_decomp_spend_dist_positive_coefficients(self) -> None:
        # Mock Ridge model with positive coefficients
        mock_model = Ridge()
        mock_model.coef_ = np.array([0.5, 1.2, 2.3])

        # Create DataFrame X with specified columns and data
        X = pd.DataFrame(
            {"media1": [1, 2, 3], "media2": [4, 5, 6], "media3": [7, 8, 9]}
        )

        # Series y with given values
        y = pd.Series([1, 2, 3])

        # Define params dictionary with relevant metrics
        params = {
            "rsq_val": 0.9,
            "rsq_test": 0.85,
            "nrmse_val": 0.1,
            "nrmse_test": 0.15,
            "nrmse": 0.05,
            "decomp_rssd": 0.02,
            "mape": 0.01,
            "lambda_": 0.5,
            "lambda_hp": 0.4,
            "lambda_max": 0.6,
            "lambda_min_ratio": 0.0001,
            "solID": "test_solID",
            "trial": 1,
            "iter_ng": 1,
            "iter_par": 1,
        }

        # Call _calculate_decomp_spend_dist
        result_df = self.ridge_model_builder._calculate_decomp_spend_dist(
            mock_model, X, y, params
        )

        # Assert that all values in the 'pos' column of the result DataFrame are True
        self.assertTrue(result_df["pos"].all())

    def test_calculate_decomp_spend_dist_mixed_coefficients(self):
        # Instantiate a Ridge model and mock its `coef_` attribute with mixed coefficients
        model = Ridge()
        model.coef_ = np.array([0.5, -0.3, 0.2, -0.1])

        # Create a DataFrame `X` with specified columns and data
        X = pd.DataFrame(
            {
                "feature1": [1, 2, 3],
                "feature2": [4, 5, 6],
                "feature3": [7, 8, 9],
                "feature4": [10, 11, 12],
            }
        )

        # Create a Series `y` with given values
        y = pd.Series([1.0, 2.0, 3.0])

        # Define the `params` dictionary with relevant metrics
        params = {
            "rsq_val": 0.85,
            "rsq_test": 0.80,
            "nrmse_val": 0.10,
            "nrmse_test": 0.15,
            "nrmse": 0.12,
            "decomp_rssd": 0.05,
            "mape": 0.07,
            "lambda_": 1.0,
            "lambda_hp": 0.5,
            "lambda_max": 2.0,
            "lambda_min_ratio": 0.0001,
            "solID": "test_model_1",
            "trial": 1,
            "iter_ng": 1,
            "iter_par": 1,
        }

        # Call `_calculate_decomp_spend_dist` with the Ridge model, DataFrame `X`, Series `y`, and dictionary `params`
        result_df = self.ridge_model_builder._calculate_decomp_spend_dist(
            model, X, y, params
        )

        # Assert that not all values in the 'pos' column of the result DataFrame are `True`
        self.assertFalse(result_df["pos"].all())

    def test_calculate_decomp_spend_dist_empty_input(self) -> None:
        # Instantiate a Ridge model and mock its coef_ attribute
        model = Ridge()
        model.coef_ = np.array([])

        # Create an empty DataFrame X and an empty Series y
        X = pd.DataFrame()
        y = pd.Series(dtype=float)

        # Define an empty params dictionary
        params = {}

        # Call _calculate_decomp_spend_dist with the Ridge model, empty DataFrame X, empty Series y, and an empty params dictionary
        result = self.ridge_model_builder._calculate_decomp_spend_dist(
            model, X, y, params
        )

        # Assert that the returned DataFrame is empty
        self.assertTrue(result.empty)

    def test_prepare_data_with_valid_data(self):
        # Mock data setup for RidgeModelBuilder
        featurized_mmm_data_mock = MagicMock()
        mmm_data_mock = MagicMock()
        featurized_mmm_data_mock.dt_mod = pd.DataFrame(
            {
                "dep_var": [1.0, 2.0, 3.0, 4.0, 5.0],
                "media1": [100, 200, 300, 400, 500],
                "media2": [50, 60, 70, 80, 90],
                "categorical_var": ["A", "B", "A", "B", "C"],
                "date_var": pd.date_range(start="2021-01-01", periods=5, freq="D"),
            }
        )
        mmm_data_mock.mmmdata_spec.dep_var = "dep_var"
        mmm_data_mock.mmmdata_spec.paid_media_spends = ["media1", "media2"]

        # Initialize RidgeModelBuilder with mock data
        ridge_model_builder = RidgeModelBuilder(
            mmm_data=mmm_data_mock,
            holiday_data=MagicMock(),
            calibration_input=MagicMock(),
            hyperparameters=MagicMock(),
            featurized_mmm_data=featurized_mmm_data_mock,
        )

        # Define parameters for the test
        params = {"media1_thetas": 0.5, "media1_alphas": 1.0, "media1_gammas": 1.0}

        # Invoke the _prepare_data method
        X, y = ridge_model_builder._prepare_data(params)

        # Validate the output
        self.assertIsInstance(X, pd.DataFrame)
        self.assertTrue(
            np.issubdtype(X.dtypes.values[0], np.number)
        )  # Ensure numeric dtype
        self.assertIsInstance(y, pd.Series)
        self.assertFalse(y.isnull().any())  # Check that NaN values are filled

    def test_prepare_data_with_nan_infinite(self):
        # Prepare mock data with NaN and infinite values
        featurized_data = pd.DataFrame(
            {
                "media1": [1.0, 2.0, np.nan, 4.0, np.inf],
                "media2": [np.nan, 2.0, 3.0, np.nan, np.inf],
                "dep_var": [1.0, np.inf, 3.0, 4.0, np.nan],
            }
        )
        self.ridge_model_builder.featurized_mmm_data.dt_mod = featurized_data
        self.ridge_model_builder.mmm_data.mmmdata_spec.dep_var = "dep_var"

        # Setup parameters
        params = {"media2_thetas": 0.5}

        # Call _prepare_data
        X, y = self.ridge_model_builder._prepare_data(params)

        # Assert that X and y have no NaN or infinite values
        self.assertFalse(np.any(np.isnan(X)))
        self.assertFalse(np.any(np.isinf(X)))
        self.assertFalse(np.any(np.isnan(y)))
        self.assertFalse(np.any(np.isinf(y)))

    def test_prepare_data_missing_dependent_var(self):
        # Mock data setup
        mock_mmm_data = Mock()
        mock_mmm_data.mmmdata_spec.dep_var = "sales"
        mock_featurized_mmm_data = Mock()
        mock_featurized_mmm_data.dt_mod = pd.DataFrame(
            {
                "media1": [1, 2, 3],
                "media2": [4, 5, 6],
                "date": pd.to_datetime(["2021-01-01", "2021-01-02", "2021-01-03"]),
            }
        )

        # Instantiate RidgeModelBuilder with mock data
        ridge_model_builder = RidgeModelBuilder(
            mmm_data=mock_mmm_data,
            holiday_data=Mock(),
            calibration_input=Mock(),
            hyperparameters=Mock(),
            featurized_mmm_data=mock_featurized_mmm_data,
        )

        # Sample parameters for testing
        params = {"media3_alphas": 0.5, "media3_gammas": 1.0}

        # Execute _prepare_data
        X, y = ridge_model_builder._prepare_data(params)

        # Verify that X does not include any column originally intended as the dependent variable
        self.assertNotIn("sales", X.columns)

        # Check that y is a Series filled with default values
        self.assertTrue(y.isna().all() or (y == y.mean()).all())

    def test_prepare_data_with_categorical_data(self):
        # Mock data setup
        mmm_data = MagicMock(spec=MMMData)
        holiday_data = MagicMock(spec=HolidaysData)
        calibration_input = MagicMock(spec=CalibrationInput)
        hyperparameters = MagicMock(spec=Hyperparameters)
        featurized_mmm_data = MagicMock(spec=FeaturizedMMMData)

        # Mock featurized_mmm_data with categorical data
        featurized_mmm_data.dt_mod = pd.DataFrame(
            {
                "category_col": ["A", "B", "A", "C"],
                "numeric_col": [1.0, 2.0, 3.0, 4.0],
                "dep_var": [10, 15, 10, 20],
            }
        )
        mmm_data.mmmdata_spec.dep_var = "dep_var"

        # Initialize RidgeModelBuilder with mock data
        ridge_model_builder = RidgeModelBuilder(
            mmm_data=mmm_data,
            holiday_data=holiday_data,
            calibration_input=calibration_input,
            hyperparameters=hyperparameters,
            featurized_mmm_data=featurized_mmm_data,
        )

        # Define params with transformation keys
        params = {"media4_thetas": 0.5}

        # Call the method to be tested
        X, y = ridge_model_builder._prepare_data(params)

        # Check that categorical data was one-hot encoded
        self.assertTrue("category_col_B" in X.columns)
        self.assertTrue("category_col_C" in X.columns)
        self.assertFalse("category_col" in X.columns)

        # Validate that y has NaN values filled accordingly
        self.assertFalse(y.isnull().any())

    def test_prepare_data_with_date_columns(self):
        # Prepare mock data for testing
        featurized_mmm_data = FeaturizedMMMData()
        featurized_mmm_data.dt_mod = pd.DataFrame(
            {
                "date": pd.to_datetime(["2021-01-01", "2021-01-02", "2021-01-03"]),
                "feature1": [1, 2, 3],
                "feature2": [4, 5, np.nan],
            }
        )

        mmm_data = MMMData()
        mmm_data.mmmdata_spec = type("", (), {})()
        mmm_data.mmmdata_spec.dep_var = "feature2"
        mmm_data.mmmdata_spec.paid_media_spends = []

        ridge_model_builder = RidgeModelBuilder(
            mmm_data=mmm_data,
            holiday_data=None,
            calibration_input=None,
            hyperparameters=None,
            featurized_mmm_data=featurized_mmm_data,
        )

        params = {"media5_alphas": 0.5, "media5_gammas": 0.5}
        X, y = ridge_model_builder._prepare_data(params)

        # Verify that date columns are converted to numeric format
        self.assertTrue("date" in X.columns)
        self.assertTrue(np.issubdtype(X["date"].dtype, np.number))

        # Verify that y is a Series with filled NaN values
        self.assertIsInstance(y, pd.Series)
        self.assertFalse(y.isna().any())

    def test_geometric_adstock_basic_regular_input(self) -> None:
        x = pd.Series([1, 2, 3, 4, 5])
        theta = 0.5
        expected_output = pd.Series([1, 2.5, 4.25, 6.125, 8.0625])

        y = self.ridge_model_builder._geometric_adstock(x, theta)

        pd.testing.assert_series_equal(y, expected_output)

    def test_geometric_adstock_empty_series(self) -> None:
        x = pd.Series(dtype=float)  # Initialize an empty pandas Series
        theta = 0.5  # Set the adstock decay parameter

        # Call the _geometric_adstock method
        y = self.ridge_model_builder._geometric_adstock(x, theta)

        # Assert that the output series is also empty
        pd.testing.assert_series_equal(y, pd.Series(dtype=float))

    def test_geometric_adstock_theta_zero(self) -> None:
        x = pd.Series([1, 2, 3, 4, 5])
        theta = 0.0
        y = self.ridge_model_builder._geometric_adstock(x, theta)
        pd.testing.assert_series_equal(y, x, check_exact=True)

    def test_geometric_adstock_theta_one(self) -> None:
        x = pd.Series([1, 2, 3, 4, 5])
        theta = 1.0
        expected_output = pd.Series([1, 3, 6, 10, 15])

        result = self.ridge_model_builder._geometric_adstock(x, theta)

        pd.testing.assert_series_equal(result, expected_output)

    def test_geometric_adstock_negative_values(self) -> None:
        x = pd.Series([-1, -2, -3, -4, -5])
        theta = 0.5
        expected_output = pd.Series([-1, -2.5, -4.25, -6.125, -8.0625])

        y = RidgeModelBuilder._geometric_adstock(self, x, theta)

        pd.testing.assert_series_equal(y, expected_output)

    def test_geometric_adstock_large_theta(self) -> None:
        x = pd.Series([1, 2, 3, 4, 5])
        theta = 10.0
        expected_output = pd.Series([1, 12, 123, 1234, 12345])

        result = self.ridge_model_builder._geometric_adstock(x, theta)

        pd.testing.assert_series_equal(result, expected_output, check_dtype=False)

    def test_geometric_adstock_small_theta(self) -> None:
        x = pd.Series([1, 2, 3, 4, 5])
        theta = 1e-09
        y = self.ridge_model_builder._geometric_adstock(x, theta)
        expected_output = pd.Series(
            [1, 2.000000001, 3.000000002, 4.000000003, 5.000000004]
        )
        pd.testing.assert_series_equal(
            y, expected_output, check_exact=False, rtol=1e-9, atol=1e-9
        )

    def test_hill_transformation_typical_values(self):
        # Instantiate the RidgeModelBuilder class (mock dependencies if needed)
        mock_mmm_data = MMMData()  # Mock or use appropriate constructor arguments
        mock_holiday_data = (
            HolidaysData()
        )  # Mock or use appropriate constructor arguments
        mock_calibration_input = (
            CalibrationInput()
        )  # Mock or use appropriate constructor arguments
        mock_hyperparameters = (
            Hyperparameters()
        )  # Mock or use appropriate constructor arguments
        mock_featurized_mmm_data = (
            FeaturizedMMMData()
        )  # Mock or use appropriate constructor arguments

        ridge_model_builder = RidgeModelBuilder(
            mock_mmm_data,
            mock_holiday_data,
            mock_calibration_input,
            mock_hyperparameters,
            mock_featurized_mmm_data,
        )

        # Define the test input
        x = pd.Series([0, 0.5, 1])
        alpha = 2
        gamma = 1

        # Call the method
        transformed_values = ridge_model_builder._hill_transformation(x, alpha, gamma)

        # Define the expected output
        expected_values = pd.Series([0, 0.25, 0.5])

        # Assert the transformed values match the expected values
        pd.testing.assert_series_equal(transformed_values, expected_values)

    def test_hill_transformation_zero_values(self):
        # Instantiate the RidgeModelBuilder class for testing (mock necessary dependencies)
        ridge_model_builder = RidgeModelBuilder(
            mmm_data=MagicMock(),
            holiday_data=MagicMock(),
            calibration_input=MagicMock(),
            hyperparameters=MagicMock(),
            featurized_mmm_data=MagicMock(),
        )

        # Input parameters
        x = pd.Series([0, 0.5, 1])
        alpha = 0.0
        gamma = 0.0

        # Call the _hill_transformation method
        transformed_series = ridge_model_builder._hill_transformation(x, alpha, gamma)

        # Verify the transformation results
        expected_values = pd.Series([0.5, 0.5, 0.5])  # All values should be 0.5
        pd.testing.assert_series_equal(transformed_series, expected_values)

    def test_hill_transformation_negative_values(self):
        # Create an instance of RidgeModelBuilder with mock parameters
        ridge_model_builder = RidgeModelBuilder(
            mmm_data=MagicMock(),
            holiday_data=MagicMock(),
            calibration_input=MagicMock(),
            hyperparameters=MagicMock(),
            featurized_mmm_data=MagicMock(),
        )

        # Define the test input for the hill transformation
        x = pd.Series([0, 0.5, 1])
        alpha = -1
        gamma = -1

        # Perform the hill transformation using negative alpha and gamma
        result = ridge_model_builder._hill_transformation(x, alpha, gamma)

        # Assert that output values are negative
        self.assertTrue(
            (result < 0).all(),
            "All transformed values should be negative with negative alpha and gamma",
        )

    def test_hill_transformation_large_series(self):
        # Given
        x = pd.Series(np.linspace(0, 1, 10000))
        alpha = 1.5
        gamma = 0.5

        # Instantiate RidgeModelBuilder with mock parameters
        ridge_model_builder = RidgeModelBuilder(
            mmm_data=Mock(spec=MMMData),
            holiday_data=Mock(spec=HolidaysData),
            calibration_input=Mock(spec=CalibrationInput),
            hyperparameters=Mock(spec=Hyperparameters),
            featurized_mmm_data=Mock(spec=FeaturizedMMMData),
        )

        # When
        transformed_series = ridge_model_builder._hill_transformation(x, alpha, gamma)

        # Then
        # Check that the transformation results are within expected bounds
        self.assertTrue(
            (transformed_series >= 0).all() and (transformed_series <= 1).all()
        )
        # Optionally, further checks can be added based on the known behavior of the hill transformation

    def test_hill_transformation_uniform_values(self):
        # Instantiate the RidgeModelBuilder with mock data
        ridge_model_builder = RidgeModelBuilder(
            mmm_data=MockMMMData(),
            holiday_data=MockHolidaysData(),
            calibration_input=MockCalibrationInput(),
            hyperparameters=MockHyperparameters(),
            featurized_mmm_data=MockFeaturizedMMMData(),
        )

        # Uniform input Series
        x = pd.Series([0.5, 0.5, 0.5, 0.5, 0.5])
        alpha = 2
        gamma = 1

        # Call the _hill_transformation method
        transformed_x = ridge_model_builder._hill_transformation(x, alpha, gamma)

        # Assert all transformed values are identical
        self.assertTrue(
            (transformed_x == transformed_x.iloc[0]).all(),
            "All transformed values should be equal.",
        )

    def test_evaluate_model_with_time_series_validation_and_positive_objective_weights(
        self,
    ) -> None:
        # Mock the _prepare_data method to return predefined X and y
        self.ridge_model_builder._prepare_data = lambda params: (
            pd.DataFrame({"feature1": [1, 2, 3, 4, 5], "feature2": [5, 4, 3, 2, 1]}),
            pd.Series([1, 2, 3, 4, 5]),
        )

        # Mock the _calculate_rssd method to return 0.5
        self.ridge_model_builder._calculate_rssd = lambda coefs, rssd_zero_penalty: 0.5

        # Mock the calibrate method of MediaEffectCalibrator to return an object with get_mean_mape
        mock_calibrator = MediaEffectCalibrator(None, None, None)
        mock_calibrator.calibrate = lambda: type(
            "obj", (object,), {"get_mean_mape": lambda: 0.05}
        )()
        self.ridge_model_builder.calibration_engine = mock_calibrator

        # Mock the _calculate_decomp_spend_dist method
        mock_decomp_spend_dist = pd.DataFrame(
            {
                "rn": ["feature1", "feature2"],
                "coef": [0.1, 0.2],
                "xDecompAgg": [0.1, 0.9],
            }
        )
        self.ridge_model_builder._calculate_decomp_spend_dist = (
            lambda model, X, y, params: mock_decomp_spend_dist
        )

        # Mock the _calculate_x_decomp_agg method
        mock_x_decomp_agg = pd.DataFrame(
            {"rn": ["feature1", "feature2"], "xDecompAgg": [0.1, 0.9]}
        )
        self.ridge_model_builder._calculate_x_decomp_agg = (
            lambda model, X, y, params: mock_x_decomp_agg
        )

        # Prepare input parameters
        params = {"train_size": 0.8, "lambda": 0.5}
        ts_validation = True
        add_penalty_factor = False
        rssd_zero_penalty = True
        objective_weights = [0.1, 0.1, 0.1]
        start_time = time.time()
        iter_ng = 1
        trial = 1

        # Call the function
        result = self.ridge_model_builder._evaluate_model(
            params,
            ts_validation,
            add_penalty_factor,
            rssd_zero_penalty,
            objective_weights,
            start_time,
            iter_ng,
            trial,
        )

        # Assertions
        self.assertEqual(result["loss"], 0.2)
        self.assertIsInstance(result["params"]["rsq_train"], float)
        self.assertIsInstance(result["params"]["rsq_val"], float)
        self.assertIsInstance(result["params"]["rsq_test"], float)
        self.assertIsInstance(result["params"]["nrmse_train"], float)
        self.assertIsInstance(result["params"]["nrmse_val"], float)
        self.assertIsInstance(result["params"]["nrmse_test"], float)
        self.assertIsInstance(result["params"]["nrmse"], float)
        self.assertEqual(result["params"]["decomp.rssd"], 0.5)
        self.assertEqual(result["params"]["mape"], 0.05)
        self.assertEqual(result["params"]["lambda"], 0.5)
        self.assertEqual(result["params"]["solID"], "1_2_1")
        self.assertEqual(result["params"]["trial"], 1)
        self.assertEqual(result["params"]["iterNG"], 2)
        self.assertEqual(result["params"]["iterPar"], 1)
        self.assertEqual(result["params"]["train_size"], 0.8)
        pd.testing.assert_frame_equal(
            result["decomp_spend_dist"], mock_decomp_spend_dist
        )
        pd.testing.assert_frame_equal(result["x_decomp_agg"], mock_x_decomp_agg)

    def test_evaluate_model_without_time_series_validation_and_negative_objective_weights(
        self,
    ):
        # Mock the _prepare_data method
        self.ridge_model_builder._prepare_data = lambda params: (
            pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]}),
            pd.Series([7, 8, 9]),
        )

        # Mock the _calculate_rssd method
        self.ridge_model_builder._calculate_rssd = lambda coefs, rssd_zero_penalty: 0.7

        # Prepare test input parameters
        params = {"train_size": 1.0, "lambda": 0.2}
        ts_validation = False
        add_penalty_factor = False
        rssd_zero_penalty = False
        objective_weights = [-1.0, -0.5, 0.0]
        start_time = time.time()
        iter_ng = 2
        trial = 2

        # Call the _evaluate_model function
        result = self.ridge_model_builder._evaluate_model(
            params,
            ts_validation,
            add_penalty_factor,
            rssd_zero_penalty,
            objective_weights,
            start_time,
            iter_ng,
            trial,
        )

        # Assertions
        self.assertEqual(result["loss"], -0.35)
        self.assertIsInstance(result["params"]["rsq_train"], float)
        self.assertIsInstance(result["params"]["nrmse_train"], float)
        self.assertIsInstance(result["params"]["nrmse"], float)
        self.assertEqual(result["params"]["rsq_val"], 0.0)
        self.assertEqual(result["params"]["rsq_test"], 0.0)
        self.assertEqual(result["params"]["nrmse_val"], 0.0)
        self.assertEqual(result["params"]["nrmse_test"], 0.0)
        self.assertEqual(result["params"]["decomp.rssd"], 0.7)
        self.assertEqual(result["params"]["mape"], 0.0)
        self.assertEqual(result["params"]["lambda"], 0.2)
        self.assertEqual(result["params"]["solID"], "2_3_1")
        self.assertEqual(result["params"]["trial"], 2)
        self.assertEqual(result["params"]["iterNG"], 3)
        self.assertEqual(result["params"]["iterPar"], 1)
        self.assertEqual(result["params"]["train_size"], 1.0)

    def test_hyper_collector_all_parameters_fixed(self) -> None:
        # Prepare input dictionary with all hyperparameters fixed
        hyperparameters_dict = {
            "prepared_hyperparameters": {
                "media1": {"thetas": 0.5, "alphas": 0.3, "gammas": 0.7},
                "media2": {"thetas": 0.6, "alphas": 0.4, "gammas": 0.8},
            },
            "hyper_to_optimize": [],
        }

        dt_hyper_fixed = pd.DataFrame(
            {
                "media1_thetas": [0.5],
                "media1_alphas": [0.3],
                "media1_gammas": [0.7],
                "media2_thetas": [0.6],
                "media2_alphas": [0.4],
                "media2_gammas": [0.8],
            }
        )

        # Call the _hyper_collector function
        result = RidgeModelBuilder._hyper_collector(
            hyperparameters_dict,
            ts_validation=False,
            add_penalty_factor=False,
            dt_hyper_fixed=dt_hyper_fixed,
            cores=4,
        )

        # Assertions
        expected_hyper_list_all = hyperparameters_dict["prepared_hyperparameters"]
        self.assertEqual(result["hyper_list_all"], expected_hyper_list_all)
        self.assertEqual(result["hyper_bound_list_updated"], {})
        self.assertEqual(
            result["hyper_bound_list_fixed"],
            {
                "media1_thetas": 0.5,
                "media1_alphas": 0.3,
                "media1_gammas": 0.7,
                "media2_thetas": 0.6,
                "media2_alphas": 0.4,
                "media2_gammas": 0.8,
            },
        )
        self.assertTrue(result["all_fixed"])

    def test_hyper_collector_some_parameters_to_optimize() -> None:
        # Construct the input dictionary with some hyperparameters set to None to specify optimization
        hyperparameters_dict = {
            "prepared_hyperparameters": {
                "hyperparameters": {
                    "channel1": {
                        "thetas": None,
                        "shapes": 1.0,
                        "scales": None,
                        "alphas": 0.1,
                        "gammas": None,
                    },
                    "channel2": {
                        "thetas": 0.5,
                        "shapes": None,
                        "scales": 1.0,
                        "alphas": None,
                        "gammas": 0.3,
                    },
                }
            },
            "hyper_to_optimize": [
                "channel1_thetas",
                "channel1_scales",
                "channel1_gammas",
                "channel2_shapes",
                "channel2_alphas",
            ],
        }

        # Invoke the _hyper_collector function
        result = RidgeModelBuilder._hyper_collector(
            hyperparameters_dict=hyperparameters_dict,
            ts_validation=True,
            add_penalty_factor=True,
            dt_hyper_fixed=None,
            cores=2,
        )

        # Assert that hyper_list_all matches the expected dictionary
        assert (
            result["hyper_list_all"]
            == hyperparameters_dict["prepared_hyperparameters"]["hyperparameters"]
        )

        # Verify that hyper_bound_list_updated contains the list of parameters set for optimization
        assert result["hyper_bound_list_updated"] == {
            "channel1_thetas": None,
            "channel1_scales": None,
            "channel1_gammas": None,
            "channel2_shapes": None,
            "channel2_alphas": None,
        }

        # Validate that hyper_bound_list_fixed includes only fixed hyperparameters
        assert result["hyper_bound_list_fixed"] == {
            "channel1_shapes": 1.0,
            "channel2_thetas": 0.5,
            "channel2_scales": 1.0,
            "channel2_gammas": 0.3,
        }

        # Confirm all_fixed is False
        assert result["all_fixed"] is False

    def test_hyper_collector_no_fixed_hyper_and_missing_params(self) -> None:
        # Prepare the input dictionary with missing hyperparameters denoted by None
        hyperparameters_dict = {
            "prepared_hyperparameters": {
                "channel1": {
                    "thetas": None,
                    "shapes": 0.5,
                    "scales": None,
                    "alphas": None,
                    "gammas": 0.3,
                    "penalty": 0.1,
                },
                "channel2": {
                    "thetas": 0.2,
                    "shapes": None,
                    "scales": 1.0,
                    "alphas": None,
                    "gammas": None,
                    "penalty": None,
                },
            },
            "hyper_to_optimize": [
                "channel1_thetas",
                "channel1_scales",
                "channel1_alphas",
                "channel2_shapes",
                "channel2_alphas",
                "channel2_gammas",
                "channel2_penalty",
            ],
        }

        # Execute the _hyper_collector function
        result = RidgeModelBuilder._hyper_collector(
            hyperparameters_dict=hyperparameters_dict,
            ts_validation=False,
            add_penalty_factor=True,
            dt_hyper_fixed=None,
            cores=1,
        )

        # Assert the hyper_list_all reflects the original hyperparameters with None for optimizable ones
        self.assertEqual(
            result["hyper_list_all"], hyperparameters_dict["prepared_hyperparameters"]
        )

        # Confirm hyper_bound_list_updated includes all parameters marked for optimization
        self.assertEqual(
            set(result["hyper_bound_list_updated"].keys()),
            set(hyperparameters_dict["hyper_to_optimize"]),
        )

        # Check that hyper_bound_list_fixed only contains parameters not set for optimization
        self.assertEqual(
            result["hyper_bound_list_fixed"],
            {"channel1_shapes": 0.5, "channel2_thetas": 0.2, "channel2_scales": 1.0},
        )

        # Verify that all_fixed is False
        self.assertFalse(result["all_fixed"])

    def test_model_refit_with_minimum_input() -> None:
        # Set up the test data
        x_train = np.random.rand(10, 5)
        y_train = np.random.rand(10)

        # Call the _model_refit function
        output = RidgeModelBuilder._model_refit(x_train, y_train)

        # Perform assertions to verify the results
        assert isinstance(output.rsq_train, float) and 0 <= output.rsq_train <= 1
        assert output.rsq_val is None
        assert output.rsq_test is None
        assert isinstance(output.nrmse_train, float) and output.nrmse_train >= 0
        assert output.nrmse_val is None
        assert output.nrmse_test is None
        assert isinstance(output.coefs, np.ndarray) and output.coefs.shape == (5,)
        assert isinstance(
            output.y_train_pred, np.ndarray
        ) and output.y_train_pred.shape == (10,)
        assert output.y_val_pred is None
        assert output.y_test_pred is None
        assert isinstance(output.y_pred, np.ndarray) and output.y_pred.shape == (10,)
        assert isinstance(output.mod, Ridge) and output.mod.alpha == 1.0
        assert output.df_int == 1

    def test_model_refit_with_full_data(self) -> None:
        # Set up the test data
        x_train = np.random.rand(10, 5)
        y_train = np.random.rand(10)
        x_val = np.random.rand(5, 5)
        y_val = np.random.rand(5)
        x_test = np.random.rand(5, 5)
        y_test = np.random.rand(5)

        # Call the _model_refit function
        output = RidgeModelBuilder._model_refit(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            x_test=x_test,
            y_test=y_test,
            lambda_=1.0,
        )

        # Perform assertions
        self.assertIsInstance(output.rsq_train, float)
        self.assertIsInstance(output.rsq_val, float)
        self.assertIsInstance(output.rsq_test, float)
        self.assertGreaterEqual(output.rsq_train, 0)
        self.assertLessEqual(output.rsq_train, 1)
        self.assertGreaterEqual(output.rsq_val, 0)
        self.assertLessEqual(output.rsq_val, 1)
        self.assertGreaterEqual(output.rsq_test, 0)
        self.assertLessEqual(output.rsq_test, 1)

        self.assertGreaterEqual(output.nrmse_train, 0)
        self.assertGreaterEqual(output.nrmse_val, 0)
        self.assertGreaterEqual(output.nrmse_test, 0)

        self.assertEqual(output.coefs.shape, (5,))
        self.assertEqual(output.y_train_pred.shape, (10,))
        self.assertEqual(output.y_val_pred.shape, (5,))
        self.assertEqual(output.y_test_pred.shape, (5,))
        self.assertEqual(output.y_pred.shape, (20,))

        self.assertIsInstance(output.mod, Ridge)
        self.assertEqual(output.mod.alpha, 1.0)
        self.assertEqual(output.df_int, 1)

    def test_model_refit_without_intercept(self):
        # Prepare mock datasets
        x_train = np.random.rand(10, 5)
        y_train = np.random.rand(10)
        x_val = np.random.rand(5, 5)
        y_val = np.random.rand(5)
        x_test = np.random.rand(5, 5)
        y_test = np.random.rand(5)

        # Call the _model_refit function with intercept set to False
        output = RidgeModelBuilder._model_refit(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            x_test=x_test,
            y_test=y_test,
            lambda_=1.0,
            intercept=False,
        )

        # Perform assertions
        self.assertTrue(0 <= output.rsq_train <= 1)
        self.assertTrue(
            0 <= output.rsq_val <= 1 if output.rsq_val is not None else True
        )
        self.assertTrue(
            0 <= output.rsq_test <= 1 if output.rsq_test is not None else True
        )
        self.assertGreaterEqual(output.nrmse_train, 0)
        self.assertGreaterEqual(
            output.nrmse_val, 0 if output.nrmse_val is not None else True
        )
        self.assertGreaterEqual(
            output.nrmse_test, 0 if output.nrmse_test is not None else True
        )

        self.assertEqual(output.coefs.shape, (5,))
        self.assertEqual(output.y_train_pred.shape, (10,))
        self.assertEqual(
            output.y_val_pred.shape, (5,) if output.y_val_pred is not None else True
        )
        self.assertEqual(
            output.y_test_pred.shape, (5,) if output.y_test_pred is not None else True
        )
        self.assertEqual(output.y_pred.shape, (20,))
        self.assertIsInstance(output.mod, Ridge)
        self.assertEqual(output.mod.alpha, 1.0)
        self.assertEqual(output.df_int, 0)

    def test_model_refit_with_zero_lambda(self) -> None:
        # Prepare datasets
        x_train = np.random.rand(10, 5)
        y_train = np.random.rand(10)
        x_val = np.random.rand(5, 5)
        y_val = np.random.rand(5)
        x_test = np.random.rand(5, 5)
        y_test = np.random.rand(5)

        # Call _model_refit with lambda_ set to 0.0
        output = RidgeModelBuilder._model_refit(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            x_test=x_test,
            y_test=y_test,
            lambda_=0.0,
        )

        # Perform assertions
        self.assertIsInstance(output.rsq_train, float)
        self.assertGreaterEqual(output.rsq_train, 0)
        self.assertLessEqual(output.rsq_train, 1)

        if output.rsq_val is not None:
            self.assertIsInstance(output.rsq_val, float)
            self.assertGreaterEqual(output.rsq_val, 0)
            self.assertLessEqual(output.rsq_val, 1)

        if output.rsq_test is not None:
            self.assertIsInstance(output.rsq_test, float)
            self.assertGreaterEqual(output.rsq_test, 0)
            self.assertLessEqual(output.rsq_test, 1)

        self.assertIsInstance(output.nrmse_train, float)
        self.assertGreaterEqual(output.nrmse_train, 0)

        if output.nrmse_val is not None:
            self.assertIsInstance(output.nrmse_val, float)
            self.assertGreaterEqual(output.nrmse_val, 0)

        if output.nrmse_test is not None:
            self.assertIsInstance(output.nrmse_test, float)
            self.assertGreaterEqual(output.nrmse_test, 0)

        self.assertEqual(output.coefs.shape, (5,))
        self.assertEqual(output.y_train_pred.shape, (10,))

        if output.y_val_pred is not None:
            self.assertEqual(output.y_val_pred.shape, (5,))

        if output.y_test_pred is not None:
            self.assertEqual(output.y_test_pred.shape, (5,))

        self.assertEqual(output.y_pred.shape, (20,))
        self.assertIsInstance(output.mod, Ridge)
        self.assertEqual(output.mod.alpha, 0.0)
        self.assertEqual(output.df_int, 1)

    def test_model_refit_with_coefficient_limits() -> None:
        # Prepare the datasets
        x_train = np.random.rand(10, 5)
        y_train = np.random.rand(10)
        x_val = np.random.rand(5, 5)
        y_val = np.random.rand(5)
        x_test = np.random.rand(5, 5)
        y_test = np.random.rand(5)

        # Define lower and upper limits for coefficients
        lower_limits = [-1, -1, -1, -1, -1]
        upper_limits = [1, 1, 1, 1, 1]

        # Call the _model_refit function
        output = RidgeModelBuilder._model_refit(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            x_test=x_test,
            y_test=y_test,
            lambda_=1.0,
            lower_limits=lower_limits,
            upper_limits=upper_limits,
            intercept=True,
            intercept_sign="non_negative",
        )

        # Perform assertions
        assert isinstance(output.rsq_train, float) and 0 <= output.rsq_train <= 1
        assert isinstance(output.rsq_val, float) and 0 <= output.rsq_val <= 1
        assert isinstance(output.rsq_test, float) and 0 <= output.rsq_test <= 1
        assert isinstance(output.nrmse_train, float) and output.nrmse_train >= 0
        assert isinstance(output.nrmse_val, float) and output.nrmse_val >= 0
        assert isinstance(output.nrmse_test, float) and output.nrmse_test >= 0
        assert (
            output.coefs.shape == (5,)
            and np.all(output.coefs >= lower_limits)
            and np.all(output.coefs <= upper_limits)
        )
        assert output.y_train_pred.shape == (10,)
        assert output.y_val_pred.shape == (5,)
        assert output.y_test_pred.shape == (5,)
        assert output.y_pred.shape == (20,)
        assert isinstance(output.mod, Ridge) and output.mod.alpha == 1.0
        assert output.df_int == 1

    def test_lambda_seq_with_small_dataset(self) -> None:
        x = np.random.rand(5, 2)
        y = np.random.rand(5)
        lambda_sequence = RidgeModelBuilder._lambda_seq(
            x, y, seq_len=10, lambda_min_ratio=0.0001
        )

        self.assertEqual(
            len(lambda_sequence), 10, "The length of the lambda sequence should be 10."
        )
        self.assertGreater(
            lambda_sequence[0],
            lambda_sequence[-1],
            "The first element should be greater than the last element.",
        )
        self.assertTrue(
            all(
                lambda_sequence[i] >= lambda_sequence[i + 1]
                for i in range(len(lambda_sequence) - 1)
            ),
            "All elements in the lambda sequence should be in non-increasing order.",
        )

    def test_lambda_seq_with_zero_feature_data(self) -> None:
        x = np.zeros((5, 2))
        y = np.random.rand(5)
        result = RidgeModelBuilder._lambda_seq(x, y, seq_len=5, lambda_min_ratio=0.0001)
        self.assertTrue(
            np.all(result == 0),
            "Expected all lambda sequence values to be zero for zero feature data.",
        )

    def test_lambda_seq_with_single_element_data(self) -> None:
        # Prepare single element input data
        x = np.random.rand(1, 1)
        y = np.random.rand(1)

        # Call the _lambda_seq function
        result = RidgeModelBuilder._lambda_seq(x, y, seq_len=5, lambda_min_ratio=0.0001)

        # Assert the length of the result is 5
        self.assertEqual(len(result), 5)

        # Assert the sequence is in descending order
        self.assertGreater(result[0], result[-1])

        # Assert all values are non-negative
        self.assertTrue(np.all(result >= 0))

    def test_lambda_seq_with_negative_values(self):
        x = np.random.uniform(-10, 0, (5, 2))
        y = np.random.uniform(-10, 10, 5)
        result = RidgeModelBuilder._lambda_seq(
            x, y, seq_len=15, lambda_min_ratio=0.0001
        )

        self.assertEqual(len(result), 15)
        self.assertGreater(result[0], result[-1])
        self.assertTrue(np.all(result >= 0))


if __name__ == '__main__':
    unittest.main()
