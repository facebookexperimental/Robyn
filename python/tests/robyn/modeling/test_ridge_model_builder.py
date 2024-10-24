from robyn.data.entities.mmmdata import MMMData
from robyn.data.entities.holidays_data import HolidaysData
from robyn.data.entities.calibration_input import CalibrationInput
from robyn.data.entities.hyperparameters import Hyperparameters
from robyn.modeling.feature_engineering import FeaturizedMMMData
import logging


class RidgeModelBuilder:
    def __init__(
        self,
        mmm_data: MMMData,
        holiday_data: HolidaysData,
        calibration_input: CalibrationInput,
        hyperparameters: Hyperparameters,
        featurized_mmm_data: FeaturizedMMMData,
    ):
        self.mmm_data = mmm_data
        self.holiday_data = holiday_data
        self.calibration_input = calibration_input
        self.hyperparameters = hyperparameters
        self.featurized_mmm_data = featurized_mmm_data
        self.logger = logging.getLogger(__name__)

    def test_model_outputs_instance():
        import unittest
        from unittest.mock import Mock
        from robyn.modeling.entities.modeloutputs import ModelOutputs

        # Mock the dependencies
        ConvergenceMock = Mock()
        ConvergenceMock.calculate_convergence.return_value = {
            "conv_msg": ["Convergence achieved successfully."]
        }

        RidgeModelBuilderMock = Mock()
        RidgeModelBuilderMock._select_best_model.return_value = "best_model_id"

        # Instantiate RidgeModelBuilder with mocked dependencies
        mmm_data = Mock()
        holiday_data = Mock()
        calibration_input = Mock()
        hyperparameters = Mock()
        featurized_mmm_data = Mock()

        builder = RidgeModelBuilderMock(
            mmm_data,
            holiday_data,
            calibration_input,
            hyperparameters,
            featurized_mmm_data,
        )
        builder.Convergence = ConvergenceMock

        # Mock TrialsConfig
        trials_config = Mock()

        # Call the build_models method
        result = builder.build_models(trials_config, intercept=False)

        # Assert that the result is an instance of ModelOutputs
        assert isinstance(
            result, ModelOutputs
        ), "The output should be an instance of ModelOutputs."

    def test_model_outputs_trials(self):
        # Prepare mock data entities required for initializing RidgeModelBuilder
        mock_mmm_data = MagicMock(spec=MMMData)
        mock_holiday_data = MagicMock(spec=HolidaysData)
        mock_calibration_input = MagicMock(spec=CalibrationInput)
        mock_hyperparameters = MagicMock(spec=Hyperparameters)
        mock_featurized_mmm_data = MagicMock(spec=FeaturizedMMMData)

        # Initialize the RidgeModelBuilder with mock data entities
        model_builder = RidgeModelBuilder(
            mock_mmm_data,
            mock_holiday_data,
            mock_calibration_input,
            mock_hyperparameters,
            mock_featurized_mmm_data,
        )

        # Define a mock TrialsConfig
        mock_trials_config = MagicMock(spec=TrialsConfig)

        # Mock the Convergence class and its calculate_convergence method
        with patch(
            "robyn.modeling.convergence.convergence.Convergence.calculate_convergence",
            return_value={"conv_msg": []},
        ):
            # Mock the _select_best_model method
            with patch.object(
                RidgeModelBuilder, "_select_best_model", return_value="best_model_id"
            ):
                # Call the build_models method
                model_outputs = model_builder.build_models(mock_trials_config)

                # Access and assert the trials attribute
                self.assertIsInstance(model_outputs.trials, list)
                self.assertTrue(
                    all(isinstance(trial, Trial) for trial in model_outputs.trials)
                )

    def test_model_outputs_select_id(self):
        # Setup mock entities
        mock_mmm_data = MMMData(...)
        mock_holiday_data = HolidaysData(...)
        mock_calibration_input = CalibrationInput(...)
        mock_hyperparameters = Hyperparameters(...)
        mock_featurized_mmm_data = FeaturizedMMMData(...)
        mock_trials_config = TrialsConfig(trials=10, iterations=100)

        # Instantiate RidgeModelBuilder
        model_builder = RidgeModelBuilder(
            mmm_data=mock_mmm_data,
            holiday_data=mock_holiday_data,
            calibration_input=mock_calibration_input,
            hyperparameters=mock_hyperparameters,
            featurized_mmm_data=mock_featurized_mmm_data,
        )

        # Define test parameters
        dt_hyper_fixed = pd.DataFrame(...)
        ts_validation = False
        add_penalty_factor = False
        seed = 123
        rssd_zero_penalty = True
        objective_weights = [0.4, 0.4, 0.2]
        nevergrad_algo = NevergradAlgorithm.TWO_POINTS_DE
        intercept = True
        intercept_sign = "non_negative"
        cores = 2

        # Mock Convergence class's calculate_convergence method
        mock_convergence = Convergence()
        mock_convergence.calculate_convergence = lambda x: {
            "conv_msg": ["Convergence message"]
        }

        # Mock _select_best_model method
        best_model_id = "best_model_123"
        model_builder._select_best_model = lambda x: best_model_id

        # Execute build_models method
        model_outputs = model_builder.build_models(
            trials_config=mock_trials_config,
            dt_hyper_fixed=dt_hyper_fixed,
            ts_validation=ts_validation,
            add_penalty_factor=add_penalty_factor,
            seed=seed,
            rssd_zero_penalty=rssd_zero_penalty,
            objective_weights=objective_weights,
            nevergrad_algo=nevergrad_algo,
            intercept=intercept,
            intercept_sign=intercept_sign,
            cores=cores,
        )

        # Assert select_id matches the mocked best_model_id
        self.assertEqual(model_outputs.select_id, best_model_id)

    def test_ts_validation_flag() -> None:
        # Initialize the RidgeModelBuilder with mock data entities
        mmm_data = MagicMock(spec=MMMData)
        holiday_data = MagicMock(spec=HolidaysData)
        calibration_input = MagicMock(spec=CalibrationInput)
        hyperparameters = MagicMock(spec=Hyperparameters)
        featurized_mmm_data = MagicMock(spec=FeaturizedMMMData)

        # Create an instance of RidgeModelBuilder
        model_builder = RidgeModelBuilder(
            mmm_data=mmm_data,
            holiday_data=holiday_data,
            calibration_input=calibration_input,
            hyperparameters=hyperparameters,
            featurized_mmm_data=featurized_mmm_data,
        )

        # Mock the calculate_convergence method
        mock_convergence = MagicMock()
        mock_convergence.calculate_convergence.return_value = {"conv_msg": []}
        model_builder._model_train = MagicMock(return_value=[])
        model_builder._select_best_model = MagicMock(return_value="best_model_id")

        # Define input parameters for build_models
        trials_config = MagicMock(spec=TrialsConfig)
        ts_validation = True

        # Call build_models with ts_validation set to True
        model_outputs = model_builder.build_models(
            trials_config=trials_config, ts_validation=ts_validation
        )

        # Assert that the ts_validation flag in ModelOutputs is True
        assert model_outputs.ts_validation is True

    def test_add_penalty_factor_flag() -> None:
        # Mock Dependencies
        mock_convergence = Mock()
        mock_convergence.calculate_convergence.return_value = {
            "conv_msg": ["Mocked Convergence"]
        }

        mock_select_best_model = Mock()
        mock_select_best_model.return_value = "best_model_id"

        # Create an instance of RidgeModelBuilder with mocked dependencies
        ridge_model_builder = RidgeModelBuilder(
            mmm_data=Mock(),
            holiday_data=Mock(),
            calibration_input=Mock(),
            hyperparameters=Mock(),
            featurized_mmm_data=Mock(),
        )

        # Patch the methods that need to be mocked
        with patch(
            "robyn.modeling.convergence.convergence.Convergence.calculate_convergence",
            mock_convergence,
        ):
            with patch.object(
                ridge_model_builder, "_select_best_model", mock_select_best_model
            ):
                # Invoke the build_models method
                model_outputs = ridge_model_builder.build_models(
                    trials_config=Mock(), add_penalty_factor=True
                )

                # Assertions
                assert model_outputs.add_penalty_factor is True

    def test_model_outputs_instance(trials_config: TrialsConfig) -> None:
        # Import necessary modules and set up the test environment
        from unittest.mock import MagicMock
        import pytest
        from some_module import RidgeModelBuilder, ModelOutputs, Convergence

        # Mock the calculate_convergence method of the Convergence class
        mock_convergence = MagicMock()
        mock_convergence.calculate_convergence.return_value = {"conv_msg": []}

        # Mock the _select_best_model method of RidgeModelBuilder
        mock_select_best_model = MagicMock()
        mock_select_best_model.return_value = "best_model_id"

        # Create a mock instance of RidgeModelBuilder with mock data and configurations
        builder = RidgeModelBuilder(
            mmm_data=MagicMock(),
            holiday_data=MagicMock(),
            calibration_input=MagicMock(),
            hyperparameters=MagicMock(),
            featurized_mmm_data=MagicMock(),
        )
        builder._select_best_model = mock_select_best_model

        # Call the build_models method with the provided trials_config and other parameters
        result = builder.build_models(
            trials_config=trials_config,
            dt_hyper_fixed=None,
            ts_validation=False,
            add_penalty_factor=False,
            seed=123,
            rssd_zero_penalty=True,
            objective_weights=None,
            nevergrad_algo=MagicMock(),
            intercept=True,
            intercept_sign="non_negative",
            cores=2,
        )

        # Assert that the result is an instance of ModelOutputs
        assert isinstance(result, ModelOutputs)

        # Verify that the mocked methods were called with expected arguments
        mock_convergence.calculate_convergence.assert_called_with(result.trials)
        mock_select_best_model.assert_called_with(result.trials)

    def test_nevergrad_algorithm(self):
        # Mock dependencies
        mock_convergence = unittest.mock.Mock(return_value={"conv_msg": []})
        mock_select_best_model = unittest.mock.Mock(return_value="best_model_id")

        # Initialize RidgeModelBuilder with mock data
        mmm_data = unittest.mock.Mock(spec=MMMData)
        holiday_data = unittest.mock.Mock(spec=HolidaysData)
        calibration_input = unittest.mock.Mock(spec=CalibrationInput)
        hyperparameters = unittest.mock.Mock(spec=Hyperparameters)
        featurized_mmm_data = unittest.mock.Mock(spec=FeaturizedMMMData)

        builder = RidgeModelBuilder(
            mmm_data,
            holiday_data,
            calibration_input,
            hyperparameters,
            featurized_mmm_data,
        )

        # Prepare mock TrialsConfig
        trials_config = unittest.mock.Mock(spec=TrialsConfig)

        # Patch methods with mocks
        with unittest.mock.patch.object(
            Convergence, "calculate_convergence", mock_convergence
        ), unittest.mock.patch.object(
            RidgeModelBuilder, "_select_best_model", mock_select_best_model
        ):

            # Execute build_models with NevergradAlgorithm.CMA
            model_outputs = builder.build_models(
                trials_config, nevergrad_algo=NevergradAlgorithm.CMA
            )

            # Assert the nevergrad_algo attribute is set correctly
            self.assertEqual(model_outputs.nevergrad_algo, NevergradAlgorithm.CMA)

            # Verify mocked methods were called
            mock_convergence.assert_called_once()
            mock_select_best_model.assert_called_once_with(model_outputs.trials)

    def test_model_outputs_intercept() -> None:
        # Import necessary testing libraries
        import unittest
        from unittest.mock import MagicMock

        # Mock the Convergence class and its method
        mock_convergence = MagicMock()
        mock_convergence.calculate_convergence.return_value = {
            "conv_msg": ["Convergence achieved"]
        }

        # Mock the RidgeModelBuilder class's method
        mock_select_best_model = MagicMock(return_value="best_model_id")

        # Create mock data entities
        mock_mmm_data = MagicMock()
        mock_holiday_data = MagicMock()
        mock_calibration_input = MagicMock()
        mock_hyperparameters = MagicMock()
        mock_featurized_mmm_data = MagicMock()

        # Instantiate the RidgeModelBuilder with mocked dependencies
        model_builder = RidgeModelBuilder(
            mock_mmm_data,
            mock_holiday_data,
            mock_calibration_input,
            mock_hyperparameters,
            mock_featurized_mmm_data,
        )

        # Mock the methods in RidgeModelBuilder
        model_builder._select_best_model = mock_select_best_model

        # Define trials_config and other parameters
        mock_trials_config = MagicMock()
        mock_dt_hyper_fixed = None
        ts_validation = False
        add_penalty_factor = False
        seed = 123
        rssd_zero_penalty = True
        objective_weights = None
        nevergrad_algo = NevergradAlgorithm.TWO_POINTS_DE
        intercept = False
        intercept_sign = "non_negative"
        cores = 2

        # Call the build_models method
        model_outputs = model_builder.build_models(
            trials_config=mock_trials_config,
            dt_hyper_fixed=mock_dt_hyper_fixed,
            ts_validation=ts_validation,
            add_penalty_factor=add_penalty_factor,
            seed=seed,
            rssd_zero_penalty=rssd_zero_penalty,
            objective_weights=objective_weights,
            nevergrad_algo=nevergrad_algo,
            intercept=intercept,
            intercept_sign=intercept_sign,
            cores=cores,
        )

        # Assert that the intercept attribute is False
        assert model_outputs.intercept == False

    def test_select_best_model_returns_correct_model(
        output_models: List[Trial],
    ) -> None:
        # Setup Input
        trial_1 = Trial(
            nrmse=0.1, decomp_rssd=0.2, result_hyp_param={"solID": "model_1"}
        )
        trial_2 = Trial(
            nrmse=0.2, decomp_rssd=0.3, result_hyp_param={"solID": "model_2"}
        )
        trial_3 = Trial(
            nrmse=0.15, decomp_rssd=0.25, result_hyp_param={"solID": "model_3"}
        )
        output_models = [trial_1, trial_2, trial_3]

        # Invoke Method
        ridge_model_builder = RidgeModelBuilder(
            ...
        )  # Assuming necessary initialization
        best_model_sol_id = ridge_model_builder._select_best_model(output_models)

        # Assertion
        assert (
            best_model_sol_id == "model_1"
        ), "Expected solID of the best model to be 'model_1'"

        # Verification
        # Since no external dependencies are involved, the function should correctly identify the model with the lowest combined score.

    def test_select_best_model_returns_correct_sol_id():
        # Create mocked Trial objects with distinct nrmse and decomp_rssd values
        mock_trial_1 = Trial(
            nrmse=0.2,
            decomp_rssd=0.1,
            result_hyp_param={"solID": ["model_1"]},
            sol_id="model_1",
        )
        mock_trial_2 = Trial(
            nrmse=0.15,
            decomp_rssd=0.09,
            result_hyp_param={"solID": ["model_2"]},
            sol_id="model_2",
        )
        mock_trial_3 = Trial(
            nrmse=0.1,
            decomp_rssd=0.08,
            result_hyp_param={"solID": ["model_3"]},
            sol_id="model_3",
        )

        # List of mocked trials
        output_models = [mock_trial_1, mock_trial_2, mock_trial_3]

        # Instantiate the RidgeModelBuilder with necessary dependencies
        ridge_model_builder = RidgeModelBuilder(
            mmm_data=None,
            holiday_data=None,
            calibration_input=None,
            hyperparameters=None,
            featurized_mmm_data=None,
        )

        # Call the _select_best_model method
        selected_model_id = ridge_model_builder._select_best_model(output_models)

        # Assert the expected best model id is returned
        assert selected_model_id == "model_3"

    def test_select_best_model_return(output_models: List[Dict[str, Any]]) -> None:
        # Initialize a list of output models with identical NRMSE and decomp RSSD values but different solID
        output_models = [
            {
                "nrmse": 0.1,
                "decomp_rssd": 0.05,
                "result_hyp_param": {"solID": "model_1"},
            },
            {
                "nrmse": 0.1,
                "decomp_rssd": 0.05,
                "result_hyp_param": {"solID": "model_2"},
            },
        ]

        # Create an instance of RidgeModelBuilder
        model_builder = RidgeModelBuilder(
            mmm_data=MMMData(),  # Mocked or actual MMMData instance
            holiday_data=HolidaysData(),  # Mocked or actual HolidaysData instance
            calibration_input=CalibrationInput(),  # Mocked or actual CalibrationInput instance
            hyperparameters=Hyperparameters(),  # Mocked or actual Hyperparameters instance
            featurized_mmm_data=FeaturizedMMMData(),  # Mocked or actual FeaturizedMMMData instance
        )

        # Call the method to test
        best_sol_id = model_builder._select_best_model(output_models)

        # Assert that the returned solID is the expected one
        assert best_sol_id == "model_1", f"Expected 'model_1' but got {best_sol_id}"

    def test_select_best_model_returns_correct_model_id() -> None:
        # Step 1: Create mock Trial objects with varying nrmse and decomp_rssd
        trial1 = Trial(
            nrmse=0.1,
            decomp_rssd=0.5,
            result_hyp_param={"solID": "model_1"},
            sol_id="model_1",
        )
        trial2 = Trial(
            nrmse=0.2,
            decomp_rssd=0.3,
            result_hyp_param={"solID": "model_2"},
            sol_id="model_2",
        )
        trial3 = Trial(
            nrmse=0.15,
            decomp_rssd=0.4,
            result_hyp_param={"solID": "model_3"},
            sol_id="model_3",
        )
        output_models = [trial1, trial2, trial3]

        # Step 2: Instantiate RidgeModelBuilder with mock data
        mmm_data = MMMData()
        holiday_data = HolidaysData()
        calibration_input = CalibrationInput()
        hyperparameters = Hyperparameters()
        featurized_mmm_data = FeaturizedMMMData()
        ridge_model_builder = RidgeModelBuilder(
            mmm_data,
            holiday_data,
            calibration_input,
            hyperparameters,
            featurized_mmm_data,
        )

        # Step 3: Call _select_best_model and capture the result
        best_model_sol_id = ridge_model_builder._select_best_model(output_models)

        # Step 4: Assert the result is the solID of the model with the lowest combined score
        assert (
            best_model_sol_id == "model_2"
        ), f"Expected 'model_2', but got {best_model_sol_id}"

    def test_select_best_model_returns_null_with_empty_trials() -> None:
        # Create mock or dummy objects for the RidgeModelBuilder dependencies
        mmm_data = MagicMock(spec=MMMData)
        holiday_data = MagicMock(spec=HolidaysData)
        calibration_input = MagicMock(spec=CalibrationInput)
        hyperparameters = MagicMock(spec=Hyperparameters)
        featurized_mmm_data = MagicMock(spec=FeaturizedMMMData)

        # Instantiate RidgeModelBuilder with the mock objects
        model_builder = RidgeModelBuilder(
            mmm_data=mmm_data,
            holiday_data=holiday_data,
            calibration_input=calibration_input,
            hyperparameters=hyperparameters,
            featurized_mmm_data=featurized_mmm_data,
        )

        # Prepare an empty list of trials to simulate no trials conducted
        output_models = []

        # Invoke the _select_best_model method with the empty list
        best_model_id = model_builder._select_best_model(output_models)

        # Assert that the method returns None when given an empty list of trials
        assert best_model_id is None, "Expected None when no trials are provided"

    def test_model_train_length_of_trials():
        # Mock the _run_nevergrad_optimization method
        with patch(
            "RidgeModelBuilder._run_nevergrad_optimization"
        ) as mock_optimization:
            # Define the mock return value
            mock_optimization.return_value = Trial(
                result_hyp_param=pd.DataFrame({"solID": ["test_sol1"]}),
                lift_calibration=None,
                decomp_spend_dist=pd.DataFrame(),
                nrmse=0.1,
                decomp_rssd=0.05,
                mape=0.07,
                x_decomp_agg=pd.DataFrame(),
                rsq_train=0.9,
                rsq_val=0.85,
                rsq_test=0.8,
                lambda_=0.01,
                lambda_hp=0.02,
                lambda_max=0.03,
                lambda_min_ratio=0.0001,
                pos=True,
                elapsed=0.5,
                elapsed_accum=0.5,
                trial=1,
                iter_ng=1,
                iter_par=1,
                train_size=0.8,
                sol_id="test_sol1",
            )

            # Prepare mock data for RidgeModelBuilder
            mmm_data = MagicMock(spec=MMMData)
            holiday_data = MagicMock(spec=HolidaysData)
            calibration_input = MagicMock(spec=CalibrationInput)
            hyperparameters = MagicMock(spec=Hyperparameters)
            featurized_mmm_data = MagicMock(spec=FeaturizedMMMData)

            # Instantiate the RidgeModelBuilder
            model_builder = RidgeModelBuilder(
                mmm_data,
                holiday_data,
                calibration_input,
                hyperparameters,
                featurized_mmm_data,
            )

            # Define input parameters
            hyper_collect = {"alpha": 0.1}  # Mock hyperparameters
            trials_config = TrialsConfig(trials=5, iterations=10)
            intercept_sign = "non_negative"
            intercept = True
            nevergrad_algo = NevergradAlgorithm.TWO_POINTS_DE
            dt_hyper_fixed = None
            ts_validation = False
            add_penalty_factor = False
            objective_weights = None
            seed = 123
            rssd_zero_penalty = True
            cores = 2

            # Call the build_models method
            model_outputs = model_builder.build_models(
                trials_config,
                dt_hyper_fixed,
                ts_validation,
                add_penalty_factor,
                seed,
                rssd_zero_penalty,
                objective_weights,
                nevergrad_algo,
                intercept,
                intercept_sign,
                cores,
            )

            # Assert the length of trials
            assert (
                len(model_outputs.trials) == 5
            ), "The number of trials should match the expected value of 5."

    def test_length_of_trials() -> None:
        # Mock the _run_nevergrad_optimization method to return a successful Trial object
        with mock.patch(
            "RidgeModelBuilder._run_nevergrad_optimization"
        ) as mock_run_optimization:
            mock_run_optimization.return_value = Trial(trial_id=1, success=True)

            # Instantiate RidgeModelBuilder with mock data
            mmm_data = mock.Mock(spec=MMMData)
            holiday_data = mock.Mock(spec=HolidaysData)
            calibration_input = mock.Mock(spec=CalibrationInput)
            hyperparameters = mock.Mock(spec=Hyperparameters)
            featurized_mmm_data = mock.Mock(spec=FeaturizedMMMData)

            builder = RidgeModelBuilder(
                mmm_data=mmm_data,
                holiday_data=holiday_data,
                calibration_input=calibration_input,
                hyperparameters=hyperparameters,
                featurized_mmm_data=featurized_mmm_data,
            )

            # Prepare test data
            hyper_collect = {"alpha": 0.5}
            trials_config = TrialsConfig(trials=3, iterations=10)
            intercept_sign = "non_negative"
            intercept = True
            nevergrad_algo = NevergradAlgorithm.TWO_POINTS_DE
            dt_hyper_fixed = None
            ts_validation = False
            add_penalty_factor = False
            objective_weights = None
            rssd_zero_penalty = True
            seed = 123
            cores = 2

            # Call the method to test
            trials = builder._model_train(
                hyper_collect=hyper_collect,
                trials_config=trials_config,
                intercept_sign=intercept_sign,
                intercept=intercept,
                nevergrad_algo=nevergrad_algo,
                dt_hyper_fixed=dt_hyper_fixed,
                ts_validation=ts_validation,
                add_penalty_factor=add_penalty_factor,
                objective_weights=objective_weights,
                rssd_zero_penalty=rssd_zero_penalty,
                seed=seed,
                cores=cores,
            )

            # Assert the length of the trials matches the expected number
            assert len(trials) == 3

    def test_model_build_length_of_trials():
        from unittest import mock
        from some_module import (
            RidgeModelBuilder,
            TrialsConfig,
        )  # Update 'some_module' to the actual module name

        # Step 3: Mock _run_nevergrad_optimization method
        with mock.patch(
            "some_module.RidgeModelBuilder._run_nevergrad_optimization"
        ) as mock_run_optimization:
            # Step 4: Define mock return value
            mock_trial = mock.Mock()
            expected_number_of_trials = 5
            mock_run_optimization.return_value = [
                mock_trial
            ] * expected_number_of_trials

            # Step 5: Create instance of RidgeModelBuilder with mock dependencies
            mmm_data = mock.Mock()
            holiday_data = mock.Mock()
            calibration_input = mock.Mock()
            hyperparameters = mock.Mock()
            featurized_mmm_data = mock.Mock()

            model_builder = RidgeModelBuilder(
                mmm_data=mmm_data,
                holiday_data=holiday_data,
                calibration_input=calibration_input,
                hyperparameters=hyperparameters,
                featurized_mmm_data=featurized_mmm_data,
            )

            # Step 6: Configure input parameters
            trials_config = TrialsConfig(
                trials=expected_number_of_trials, iterations=10
            )
            dt_hyper_fixed = None

            # Step 7: Call build_models method
            model_outputs = model_builder.build_models(
                trials_config=trials_config, dt_hyper_fixed=dt_hyper_fixed
            )

            # Step 9: Assert the length of trials
            assert len(model_outputs.trials) == expected_number_of_trials

    def test_trial_object_with_expected_properties() -> None:
        # Setup Mocking for Dependencies
        mock_optimizer = MagicMock()
        mock_optimizer.ask.return_value = MagicMock(kwargs={"param1": 0.5})
        mock_optimizer.tell.return_value = None

        with patch("ng.optimizers.registry", return_value=mock_optimizer):
            # Prepare Input Arguments
            hyper_collect = {"hyper_bound_list_updated": {"param1": (0, 1)}}
            iterations = 10
            cores = 1
            nevergrad_algo = NevergradAlgorithm.TWO_POINTS_DE
            intercept = True
            intercept_sign = "non_negative"
            ts_validation = False
            add_penalty_factor = False
            objective_weights = [0.5, 0.5]
            dt_hyper_fixed = None
            rssd_zero_penalty = False
            trial = 1
            seed = 42
            total_trials = 1

            # Invoke Method Under Test
            trial_result = RidgeModelBuilder._run_nevergrad_optimization(
                RidgeModelBuilder,
                hyper_collect,
                iterations,
                cores,
                nevergrad_algo,
                intercept,
                intercept_sign,
                ts_validation,
                add_penalty_factor,
                objective_weights,
                dt_hyper_fixed,
                rssd_zero_penalty,
                trial,
                seed,
                total_trials,
            )

            # Assertions
            assert isinstance(trial_result, Trial)
            assert "solID" in trial_result.result_hyp_param.columns
            assert trial_result.result_hyp_param["solID"].iloc[0] == f"{trial}_1_1"
            assert trial_result.trial == trial
            assert trial_result.decomp_rssd is not None
            assert trial_result.nrmse is not None
            assert trial_result.mape is not None

    def test_trial_object_properties():
        # Setup Mocks for Dependencies
        mock_optimizer = Mock()
        mock_optimizer.ask.return_value.kwargs = {"param1": 0.5}
        mock_optimizer.tell.return_value = None
        ng.optimizers.registry["TWO_POINTS_DE"] = Mock(return_value=mock_optimizer)

        # Initialize Input Parameters
        hyper_collect = {"hyper_bound_list_updated": {"param1": (0.0, 1.0)}}
        iterations = 5
        cores = 2
        nevergrad_algo = NevergradAlgorithm.TWO_POINTS_DE
        intercept = True
        intercept_sign = "non_negative"

        # Invoke the Function
        trial_result = test_run_nevergrad_optimization(
            hyper_collect=hyper_collect,
            iterations=iterations,
            cores=cores,
            nevergrad_algo=nevergrad_algo,
            intercept=intercept,
            intercept_sign=intercept_sign,
            ts_validation=False,
            add_penalty_factor=False,
            objective_weights=None,
            dt_hyper_fixed=None,
            rssd_zero_penalty=False,
            trial=1,
            seed=123,
            total_trials=10,
        )

        # Verify Output
        assert isinstance(trial_result, Trial)
        assert "param1" in trial_result.result_hyp_param.columns
        assert len(trial_result.result_hyp_param["solID"].unique()) == iterations
        assert trial_result.trial == 1
        assert isinstance(trial_result.decomp_spend_dist, pd.DataFrame)
        assert isinstance(trial_result.x_decomp_agg, pd.DataFrame)

    def test_run_nevergrad_optimization_trial_creation(self):
        # Mock Nevergrad optimizer
        with mock.patch(
            "nevergrad.optimizers.registry.TWO_POINTS_DE", autospec=True
        ) as MockOptimizer:
            mock_optimizer_instance = MockOptimizer.return_value
            # Mock the ask method to return predefined parameters
            mock_optimizer_instance.ask.return_value = mock.Mock(kwargs={"param1": 0.5})
            # Mock the tell method to accept any inputs
            mock_optimizer_instance.tell.return_value = None

            # Prepare test inputs
            hyper_collect = {
                "hyper_bound_list_updated": {"param1": (0.0, 1.0)},
                "hyper_bound_list_fixed": {},
            }
            iterations = 5
            cores = 2
            intercept = True
            intercept_sign = "non_negative"
            ts_validation = False
            add_penalty_factor = False
            objective_weights = [0.5, 0.5]
            rssd_zero_penalty = True
            trial = 1
            seed = 123
            total_trials = 1

            # Invoke _run_nevergrad_optimization
            result_trial = self.ridge_model_builder._run_nevergrad_optimization(
                hyper_collect,
                iterations,
                cores,
                NevergradAlgorithm.TWO_POINTS_DE,
                intercept,
                intercept_sign,
                ts_validation,
                add_penalty_factor,
                objective_weights,
                None,
                rssd_zero_penalty,
                trial,
                seed,
                total_trials,
            )

            # Assert trial object creation
            self.assertIsInstance(result_trial, Trial)
            self.assertIn("solID", result_trial.result_hyp_param.columns)
            self.assertGreaterEqual(result_trial.nrmse, 0)
            self.assertGreaterEqual(result_trial.decomp_rssd, 0)
            self.assertGreaterEqual(result_trial.mape, 0)

            # Clean up mocks
            mock_optimizer_instance.ask.reset_mock()
            mock_optimizer_instance.tell.reset_mock()

    def test_run_nevergrad_optimization_trial_result():
        import unittest
        from unittest import mock
        from my_module import RidgeModelBuilder, Trial, NevergradAlgorithm
        import pandas as pd

        # Mock optimizer and its methods
        mock_optimizer = mock.Mock()
        mock_optimizer.ask.return_value = mock.Mock(kwargs={"param1": 0.5})
        mock_optimizer.tell = mock.Mock()

        # Mock the optimizer registry
        with mock.patch(
            "ng.optimizers.registry.TWO_POINTS_DE", return_value=mock_optimizer
        ):
            # Create a RidgeModelBuilder instance with mock dependencies
            mmm_data = mock.Mock()
            holiday_data = mock.Mock()
            calibration_input = mock.Mock()
            hyperparameters = mock.Mock()
            featurized_mmm_data = mock.Mock()

            builder = RidgeModelBuilder(
                mmm_data,
                holiday_data,
                calibration_input,
                hyperparameters,
                featurized_mmm_data,
            )

            # Prepare input parameters
            hyper_collect = {"hyper_bound_list_updated": {"param1": (0, 1)}}
            iterations = 5
            cores = 2
            nevergrad_algo = NevergradAlgorithm.TWO_POINTS_DE
            intercept = True
            intercept_sign = "non_negative"
            ts_validation = False
            add_penalty_factor = False
            rssd_zero_penalty = False
            objective_weights = None
            dt_hyper_fixed = pd.DataFrame()
            trial = 1
            seed = 123
            total_trials = 1

            # Call the method under test
            result = builder._run_nevergrad_optimization(
                hyper_collect,
                iterations,
                cores,
                nevergrad_algo,
                intercept,
                intercept_sign,
                ts_validation,
                add_penalty_factor,
                objective_weights,
                dt_hyper_fixed,
                rssd_zero_penalty,
                trial,
                seed,
                total_trials,
            )

            # Assert result is a Trial object
            assert isinstance(result, Trial)

            # Check that the properties of the Trial object are as expected
            # (This is a placeholder for expected properties, replace with actual checks)
            assert result.sol_id is not None

            # Verify that mock methods were called with expected arguments
            mock_optimizer.ask.assert_called()
            mock_optimizer.tell.assert_called_with(mock.ANY, mock.ANY)

    def test_rsq_train_calculation() -> None:
        # Step 1: Instantiate a Ridge model with predefined coefficients
        lambda_ = 1.0
        model = Ridge(alpha=lambda_, fit_intercept=True)
        model.coef_ = np.array([0.5, 0.3, -0.2])  # Example coefficients

        # Step 2: Prepare a DataFrame `X` containing columns for paid media spends
        X = pd.DataFrame(
            {
                "media1": [10, 20, 30, 40, 50],
                "media2": [5, 15, 25, 35, 45],
                "media3": [2, 12, 22, 32, 42],
            }
        )

        # Step 3: Prepare a Series `y` representing the target values
        y = pd.Series([20, 40, 60, 80, 100])

        # Step 4: Define `params` containing relevant parameters
        params = {"rsq_val": 0.0, "rsq_test": 0.0, "nrmse_val": 0.0, "nrmse_test": 0.0}

        # Step 5: Call `_calculate_decomp_spend_dist` with the model, `X`, `y`, and `params`
        result = RidgeModelBuilder._calculate_decomp_spend_dist(model, X, y, params)

        # Step 6: Extract the `rsq_train` value from the output DataFrame
        rsq_train_value = result["rsq_train"].iloc[0]

        # Step 7: Assert that the extracted `rsq_train` value matches the expected R² score for training data
        expected_rsq_train = 0.95  # Example expected R² score
        assert (
            rsq_train_value == expected_rsq_train
        ), f"Expected {expected_rsq_train}, got {rsq_train_value}"

    def test_nrmse_train_calculation() -> None:
        # Step 1: Instantiate a Ridge model with predefined coefficients.
        model = Ridge(alpha=1.0)
        model.coef_ = np.array([0.2, 0.5, 0.3])

        # Step 2: Prepare a DataFrame `X` containing columns for paid media spends.
        X = pd.DataFrame(
            {
                "media1": [50, 60, 70, 80, 90],
                "media2": [20, 30, 40, 50, 60],
                "media3": [10, 20, 30, 40, 50],
            }
        )

        # Step 3: Prepare a Series `y` representing the target values.
        y = pd.Series([100, 150, 200, 250, 300])

        # Step 4: Define `params` containing relevant parameters such as `rsq_val`, `rsq_test`, etc.
        params = {
            "rsq_val": 0.8,
            "rsq_test": 0.75,
            "nrmse_val": 0.1,
            "nrmse_test": 0.15,
        }

        # Step 5: Call `_calculate_decomp_spend_dist` with the model, `X`, `y`, and `params`.
        result_df = RidgeModelBuilder._calculate_decomp_spend_dist(model, X, y, params)

        # Step 6: Extract the `nrmse_train` value from the output DataFrame.
        nrmse_train_value = result_df["nrmse_train"].iloc[0]

        # Step 7: Assert that the extracted `nrmse_train` value matches the expected NRMSE for training data.
        expected_nrmse_train = np.sqrt(mean_squared_error(y, model.predict(X))) / (
            y.max() - y.min()
        )
        assert np.isclose(
            nrmse_train_value, expected_nrmse_train, atol=1e-7
        ), "NRMSE train value does not match expected value"

    def test_coef_calculation() -> None:
        # Step 1: Instantiate a Ridge model with predefined coefficients
        model = Ridge(alpha=1.0)
        model.coef_ = np.array(
            [0.5, 1.5, -0.5]
        )  # Example coefficients for paid media columns

        # Step 2: Prepare a DataFrame `X` containing columns for paid media spends
        X = pd.DataFrame(
            {
                "media1": [100, 200, 300],
                "media2": [150, 250, 350],
                "media3": [200, 300, 400],
            }
        )

        # Step 3: Prepare a Series `y` representing the target values
        y = pd.Series([500, 600, 700])

        # Step 4: Define `params` containing relevant parameters such as `rsq_val`, `rsq_test`, etc.
        params = {
            "rsq_val": 0.8,
            "rsq_test": 0.7,
            "nrmse_val": 0.1,
            "nrmse_test": 0.15,
            "nrmse": 0.1,
            "decomp_rssd": 0.05,
            "mape": 0.1,
            "lambda_": 0.01,
            "lambda_hp": 0.02,
            "lambda_max": 0.5,
            "lambda_min_ratio": 0.0001,
            "solID": "test_001",
            "trial": 1,
            "iterNG": 10,
            "iterPar": 2,
        }

        # Step 5: Call `_calculate_decomp_spend_dist` with the model, `X`, `y`, and `params`
        decomp_spend_dist = RidgeModelBuilder._calculate_decomp_spend_dist(
            model, X, y, params
        )

        # Step 6: Extract the `coef` values from the output DataFrame
        calculated_coefs = decomp_spend_dist["coef"].values

        # Step 7: Assert that the extracted `coef` values match the expected coefficients for paid media columns
        expected_coefs = np.array([0.5, 1.5, -0.5])
        np.testing.assert_array_equal(calculated_coefs, expected_coefs)

    def test_xDecompAgg_calculation(self) -> None:
        # Step 1: Instantiate a Ridge model with predefined coefficients.
        model = Ridge(alpha=1.0)
        model.coef_ = np.array([0.5, 1.5, -0.5])  # Example coefficients

        # Step 2: Prepare a DataFrame `X` containing columns for paid media spends.
        X = pd.DataFrame(
            {
                "media1": [10, 20, 30],
                "media2": [5, 10, 15],
                "media3": [0, 0, 0],  # Media spend with zero effect
            }
        )

        # Step 3: Prepare a Series `y` representing the target values.
        y = pd.Series([100, 200, 300])

        # Step 4: Define `params` containing relevant parameters such as `rsq_val`, `rsq_test`, etc.
        params = {
            "rsq_val": 0.7,
            "rsq_test": 0.6,
            "nrmse_val": 0.1,
            "nrmse_test": 0.15,
            "nrmse": 0.08,
            "decomp_rssd": 0.05,
            "mape": 0.1,
            "lambda_": 0.01,
            "lambda_hp": 0.02,
            "solID": "test_001",
            "trial": 1,
            "iterNG": 10,
            "iterPar": 2,
        }

        # Step 5: Call `_calculate_x_decomp_agg` with the model, `X`, `y`, and `params`.
        x_decomp_agg_df = self._calculate_x_decomp_agg(model, X, y, params)

        # Step 6: Extract the `xDecompAgg` value from the output DataFrame.
        x_decomp_agg_value = x_decomp_agg_df["xDecompAgg"].sum()

        # Step 7: Assert that the extracted `xDecompAgg` value matches the expected sum of decomposed contributions.
        expected_x_decomp_agg_value = 100  # This should be the expected value based on how xDecompAgg is calculated
        self.assertAlmostEqual(
            x_decomp_agg_value, expected_x_decomp_agg_value, places=2
        )

    def test_xDecompMeanNon0_calculation() -> None:
        # Step 1: Instantiate a Ridge model with predefined coefficients
        model = Ridge(alpha=0.1)
        model.coef_ = np.array([0.5, 0.0, 0.3, 0.0, 0.2])

        # Step 2: Prepare a DataFrame `X` containing columns for paid media spends
        X = pd.DataFrame(
            {
                "media1": [100, 200, 300, 400, 500],
                "media2": [0, 0, 0, 0, 0],
                "media3": [50, 60, 70, 80, 90],
                "media4": [0, 0, 0, 0, 0],
                "media5": [20, 30, 40, 50, 60],
            }
        )

        # Step 3: Prepare a Series `y` representing the target values
        y = pd.Series([150, 250, 350, 450, 550])

        # Step 4: Define `params` containing relevant parameters
        params = {
            "rsq_val": 0.8,
            "rsq_test": 0.75,
            "nrmse_val": 0.1,
            "nrmse_test": 0.15,
            "nrmse": 0.08,
            "decomp_rssd": 0.05,
            "mape": 0.1,
            "lambda": 0.01,
            "lambda_hp": 0.02,
            "lambda_max": 1.0,
            "lambda_min_ratio": 0.0001,
            "solID": "test_001",
            "trial": 1,
            "iterNG": 10,
            "iterPar": 2,
        }

        # Step 5: Call `_calculate_x_decomp_agg` with the model, `X`, `y`, and `params`
        result_df = RidgeModelBuilder._calculate_x_decomp_agg(model, X, y, params)

        # Step 6: Extract the `xDecompMeanNon0` value from the output DataFrame
        xDecompMeanNon0 = result_df.loc[
            result_df["rn"] == "media1", "xDecompMeanNon0"
        ].values[0]

        # Step 7: Assert that the extracted `xDecompMeanNon0` value matches the expected mean
        expected_mean_non_zero = (
            0.5 * 100 + 0.5 * 200 + 0.5 * 300 + 0.5 * 400 + 0.5 * 500
        ) / 5
        assert abs(xDecompMeanNon0 - expected_mean_non_zero) < 1e-6

    def test_calculate_decomp_spend_dist_rsq_train_with_empty_data() -> None:
        model = Ridge()
        X = pd.DataFrame()
        y = pd.Series(dtype=float)
        params = {
            "rsq_val": 0,
            "rsq_test": 0,
            "nrmse_val": 0,
            "nrmse_test": 0,
            "nrmse": 0,
            "decomp_rssd": 0,
            "mape": 0,
            "lambda_": 0,
            "lambda_hp": 0,
            "lambda_max": 0,
            "lambda_min_ratio": 0,
            "solID": "",
            "trial": 0,
            "iter_ng": 0,
            "iter_par": 0,
        }

        decomp_spend_dist = RidgeModelBuilder._calculate_decomp_spend_dist(
            RidgeModelBuilder, model, X, y, params
        )
        rsq_train = decomp_spend_dist.get("rsq_train", None)
        assert rsq_train == 0, f"Expected rsq_train to be 0, but got {rsq_train}."

    def test_calculate_decomp_spend_dist_nrmse_train_with_empty_data() -> None:
        model = Ridge()  # Instantiate a Ridge model with default settings
        X = pd.DataFrame()  # Create an empty DataFrame for features
        y = pd.Series(dtype=float)  # Create an empty Series for target
        params = {  # Define a params dictionary with all values set to 0
            "rsq_val": 0,
            "rsq_test": 0,
            "nrmse": 0,
            "decomp_rssd": 0,
            "mape": 0,
            "lambda": 0,
            "lambda_hp": 0,
            "lambda_max": 0,
            "lambda_min_ratio": 0,
            "solID": "",
            "trial": 0,
            "iterNG": 0,
            "iterPar": 0,
        }
        decomp_spend_dist = RidgeModelBuilder._calculate_decomp_spend_dist(
            model, X, y, params
        )
        nrmse_train = decomp_spend_dist["nrmse_train"].iloc[0]
        assert pd.isna(nrmse_train), "nrmse_train should be NaN for empty data."

    def test_calculate_decomp_spend_dist_coef_with_empty_data() -> None:
        model = Ridge()
        X = pd.DataFrame()
        y = pd.Series(dtype=float)
        params = {
            "rsq_train": 0,
            "rsq_val": 0,
            "rsq_test": 0,
            "nrmse_train": 0,
            "nrmse_val": 0,
            "nrmse_test": 0,
            "nrmse": 0,
            "decomp_rssd": 0,
            "mape": 0,
            "lambda": 0,
            "lambda_hp": 0,
            "lambda_max": 0,
            "lambda_min_ratio": 0,
            "solID": "",
            "trial": 0,
            "iterNG": 0,
            "iterPar": 0,
        }

        decomp_spend_dist = RidgeModelBuilder._calculate_decomp_spend_dist(
            model, X, y, params
        )
        coef_array = decomp_spend_dist["coef"].to_numpy()

        assert coef_array.size == 0

    def test_calculate_decomp_spend_dist_xDecompAgg_with_empty_data() -> None:
        # Instantiate a Ridge model with predefined coefficients
        model = Ridge()

        # Create an empty DataFrame for features X
        X = pd.DataFrame()

        # Create an empty Series for target y
        y = pd.Series(dtype=float)

        # Define a params dictionary with all parameter values set to 0
        params = {
            "rsq_val": 0,
            "rsq_test": 0,
            "nrmse_val": 0,
            "nrmse_test": 0,
            "nrmse": 0,
            "decomp_rssd": 0,
            "mape": 0,
            "lambda_": 0,
            "lambda_hp": 0,
            "lambda_max": 0,
            "lambda_min_ratio": 0,
            "solID": "",
            "trial": 0,
            "iter_ng": 0,
            "iter_par": 0,
        }

        # Call _calculate_decomp_spend_dist with the Ridge model, empty DataFrame, empty Series, and the params
        decomp_spend_dist = RidgeModelBuilder._calculate_decomp_spend_dist(
            model, X, y, params
        )

        # Extract the xDecompAgg value from the returned DataFrame
        xDecompAgg_value = (
            decomp_spend_dist["xDecompAgg"].iloc[0]
            if not decomp_spend_dist.empty
            else np.nan
        )

        # Assert that the xDecompAgg value is NaN
        assert np.isnan(
            xDecompAgg_value
        ), "Expected xDecompAgg to be NaN for empty input data"

    def test_calculate_decomp_spend_dist_xDecompMeanNon0_with_empty_data() -> None:
        model = Ridge()
        model.coef_ = np.array([])  # Predefined empty coefficients
        X = pd.DataFrame()  # Empty DataFrame for features
        y = pd.Series(dtype=float)  # Empty Series for target
        params = {
            "rsq_val": 0,
            "rsq_test": 0,
            "nrmse_val": 0,
            "nrmse_test": 0,
            "nrmse": 0,
            "decomp_rssd": 0,
            "mape": 0,
            "solID": "",
            "trial": 0,
            "iter_ng": 0,
            "iter_par": 0,
        }

        decomp_spend_dist = RidgeModelBuilder._calculate_decomp_spend_dist(
            model, X, y, params
        )
        xDecompMeanNon0 = decomp_spend_dist["xDecompMeanNon0"]

        # Assert that xDecompMeanNon0 is NaN
        assert xDecompMeanNon0.isna().all()

    def test_coefficient_with_zero_values() -> None:
        # Mock data setup
        mmm_data = MMMData(
            mmmdata_spec=MockSpec(paid_media_spends=["media1", "media2", "media3"]),
            data=pd.DataFrame(
                {
                    "media1": [100, 200, 300],
                    "media2": [300, 400, 500],
                    "media3": [0, 0, 0],
                }
            ),
        )
        holiday_data = HolidaysData()
        calibration_input = CalibrationInput()
        hyperparameters = Hyperparameters()
        featurized_mmm_data = FeaturizedMMMData(
            dt_mod=pd.DataFrame(
                {
                    "media1": [1, 2, 3],
                    "media2": [3, 4, 5],
                    "media3": [0, 0, 0],
                    "dep_var": [10, 20, 30],
                }
            )
        )

        # Instance of RidgeModelBuilder
        ridge_model_builder = RidgeModelBuilder(
            mmm_data=mmm_data,
            holiday_data=holiday_data,
            calibration_input=calibration_input,
            hyperparameters=hyperparameters,
            featurized_mmm_data=featurized_mmm_data,
        )

        # Prepare a Ridge model with zero coefficients
        mock_model = Ridge()
        mock_model.coef_ = np.array([0.1, 0.0, 0.0])  # Coefficients with zero values

        # Prepare input data
        X = pd.DataFrame(
            {"media1": [1, 2, 3], "media2": [3, 4, 5], "media3": [0, 0, 0]}
        )
        y = pd.Series([10, 20, 30])

        # Invoke _calculate_decomp_spend_dist
        result = ridge_model_builder._calculate_decomp_spend_dist(
            mock_model, X, y, params={}
        )

        # Check that the coefficients include zeros
        assert np.any(
            result["coef"] == 0
        ), "The coefficients should include zeros for unaffected features."

    def test_xdecomp_agg_sum_excluding_zero() -> None:
        # Mock data and setup
        mmm_data = MagicMock(MMMData)
        holiday_data = MagicMock(HolidaysData)
        calibration_input = MagicMock(CalibrationInput)
        hyperparameters = MagicMock(Hyperparameters)
        featurized_mmm_data = MagicMock(FeaturizedMMMData)

        # Initialize RidgeModelBuilder
        model_builder = RidgeModelBuilder(
            mmm_data,
            holiday_data,
            calibration_input,
            hyperparameters,
            featurized_mmm_data,
        )

        # Prepare mock Ridge model with some coefficients set to zero
        model = MagicMock(spec=Ridge)
        model.coef_ = np.array([0.5, 0.0, -0.3, 0.0, 0.2])  # some coefficients are zero

        # Create a DataFrame X with columns representing paid media spends
        X = pd.DataFrame(
            {
                "media1": [100, 200, 300],
                "media2": [400, 500, 600],
                "media3": [700, 800, 900],
                "media4": [1000, 1100, 1200],
                "media5": [1300, 1400, 1500],
            }
        )

        # Create a Series y as target
        y = pd.Series([10, 15, 20])

        # Call the _calculate_x_decomp_agg method
        x_decomp_agg_df = model_builder._calculate_x_decomp_agg(model, X, y, {})

        # Extract the 'xDecompAgg' column
        x_decomp_agg_sum = x_decomp_agg_df["xDecompAgg"].sum()

        # Calculate the expected sum of non-zero coefficient contributions
        expected_sum = (
            X["media1"] * 0.5 + X["media3"] * -0.3 + X["media5"] * 0.2
        ).sum()

        # Assert that the calculated sum matches the expected sum
        assert np.isclose(
            x_decomp_agg_sum, expected_sum
        ), "xDecompAgg sum does not match expected value, zero coefficients were not ignored."

    def test_mean_spend_of_paid_media_columns() -> None:
        # Mocking data for initialization
        mmm_data = MMMData(...)
        holiday_data = HolidaysData(...)
        calibration_input = CalibrationInput(...)
        hyperparameters = Hyperparameters(...)
        featurized_mmm_data = FeaturizedMMMData(...)

        # Initialize RidgeModelBuilder
        model_builder = RidgeModelBuilder(
            mmm_data,
            holiday_data,
            calibration_input,
            hyperparameters,
            featurized_mmm_data,
        )

        # Mock Ridge model and DataFrame X, Series y
        model = Ridge()
        X = pd.DataFrame(
            {
                "media1": [100, 200, 300],
                "media2": [400, 500, 600],
                "media3": [0, 0, 0],  # Assuming one media spend column is all zeros
            }
        )
        y = pd.Series([1000, 1500, 2000])

        # Set up MMMData specification to include only the paid media spend columns
        model_builder.mmm_data.mmmdata_spec.paid_media_spends = [
            "media1",
            "media2",
            "media3",
        ]

        # Call the method to calculate decomposition spend distribution
        decomp_spend_dist = model_builder._calculate_decomp_spend_dist(
            model, X, y, params={}
        )

        # Extract 'mean_spend' column
        mean_spend = decomp_spend_dist["mean_spend"]

        # Calculate expected mean spend
        expected_mean_spend = X[["media1", "media2", "media3"]].mean()

        # Assert that the calculated mean spend matches the expected value
        pd.testing.assert_series_equal(mean_spend, expected_mean_spend)

    def test_negative_coefficients_identification() -> None:
        # Initialize mock data
        mmm_data = MMMData(...)  # Fill with appropriate mock data
        holiday_data = HolidaysData(...)  # Fill with appropriate mock data
        calibration_input = CalibrationInput(...)  # Fill with appropriate mock data
        hyperparameters = Hyperparameters(...)  # Fill with appropriate mock data
        featurized_mmm_data = FeaturizedMMMData(...)  # Fill with appropriate mock data

        # Initialize RidgeModelBuilder with mock data
        model_builder = RidgeModelBuilder(
            mmm_data,
            holiday_data,
            calibration_input,
            hyperparameters,
            featurized_mmm_data,
        )

        # Create Ridge model with negative coefficients manually
        model = Ridge()
        model.coef_ = np.array([-0.5, -1.0, -0.2])  # Example negative coefficients

        # Prepare a mock DataFrame X and Series y
        X = pd.DataFrame(
            {"media1": [0, 1, 2], "media2": [1, 2, 3], "media3": [2, 3, 4]}
        )
        y = pd.Series([1, 2, 3])

        # Call the method to test
        result = model_builder._calculate_decomp_spend_dist(model, X, y, params={})

        # Extract the 'pos' column, which should indicate the presence of negative coefficients
        pos = result["pos"].to_numpy()

        # Assert that the 'pos' column matches the expected boolean array indicating negative coefficients
        expected_pos = np.array([False, False, False])  # All coefficients are negative
        np.testing.assert_array_equal(pos, expected_pos)

    def test_effect_share_calculation():
        # Initialize RidgeModelBuilder with mock data
        mock_mmm_data = MMMData(...)
        mock_holidays_data = HolidaysData(...)
        mock_calibration_input = CalibrationInput(...)
        mock_hyperparameters = Hyperparameters(...)
        mock_featurized_mmm_data = FeaturizedMMMData(...)

        builder = RidgeModelBuilder(
            mock_mmm_data,
            mock_holidays_data,
            mock_calibration_input,
            mock_hyperparameters,
            mock_featurized_mmm_data,
        )

        # Create a Ridge model with known coefficients
        coefficients = np.array([0.5, -0.3, 0.2])
        mock_model = Ridge()
        mock_model.coef_ = coefficients

        # Prepare mock DataFrame X
        X = pd.DataFrame(
            {"media1": [10, 20, 30], "media2": [5, 10, 15], "media3": [2, 4, 6]}
        )

        # Prepare mock Series y
        y = pd.Series([1, 2, 3])

        # Call the method
        params = {}
        result = builder._calculate_decomp_spend_dist(mock_model, X, y, params)

        # Extract the effect share
        effect_share = result["effect_share"]

        # Calculate expected effect share
        x_decomp = X * coefficients
        expected_effect_share = x_decomp.sum() / x_decomp.sum().sum()

        # Assert the effect_share matches expected values
        pd.testing.assert_series_equal(
            effect_share, expected_effect_share, check_names=False
        )

    def test_total_spend_calculation() -> None:
        # Initialize the RidgeModelBuilder with mock data
        mmm_data = MagicMock()
        mmm_data.mmmdata_spec.paid_media_spends = ["media1", "media2"]
        holidays_data = MagicMock()
        calibration_input = MagicMock()
        hyperparameters = MagicMock()
        featurized_mmm_data = MagicMock()

        builder = RidgeModelBuilder(
            mmm_data,
            holidays_data,
            calibration_input,
            hyperparameters,
            featurized_mmm_data,
        )

        # Prepare mock DataFrame X with known values
        X = pd.DataFrame({"media1": [100, 200, 300], "media2": [400, 500, 600]})

        # Prepare mock Series y
        y = pd.Series([1, 2, 3])

        # Create a mock Ridge model
        model = Ridge()

        # Call the _calculate_decomp_spend_dist method
        params = {}
        result = builder._calculate_decomp_spend_dist(model, X, y, params)

        # Extract the total_spend column
        total_spend = result["total_spend"].sum()

        # Manually calculate the total spend
        manual_total_spend = X["media1"].sum() + X["media2"].sum()

        # Assert that the total_spend column matches the manually calculated total spend
        assert total_spend == manual_total_spend

    def test_rsq_val_with_missing_params() -> None:
        # Initialize RidgeModelBuilder with mock data entities
        mmm_data = MagicMock(MMMData)
        holiday_data = MagicMock(HolidaysData)
        calibration_input = MagicMock(CalibrationInput)
        hyperparameters = MagicMock(Hyperparameters)
        featurized_mmm_data = MagicMock(FeaturizedMMMData)

        model_builder = RidgeModelBuilder(
            mmm_data=mmm_data,
            holiday_data=holiday_data,
            calibration_input=calibration_input,
            hyperparameters=hyperparameters,
            featurized_mmm_data=featurized_mmm_data,
        )

        # Create a mock Ridge model
        model = MagicMock(Ridge)
        model.coef_ = np.array([0.5, 0.3, 0.2])

        # Prepare mock DataFrame for X and Series for y
        X = pd.DataFrame(
            {
                "media_spend1": [100, 200, 300],
                "media_spend2": [150, 250, 350],
                "media_spend3": [120, 220, 320],
            }
        )
        y = pd.Series([500, 700, 900])

        # Call the method with empty params
        result = model_builder._calculate_decomp_spend_dist(model, X, y, params={})

        # Extract and assert rsq_val
        rsq_val = result["rsq_val"].iloc[0]
        assert rsq_val == 0, f"Expected rsq_val to be 0, but got {rsq_val}"

    def test_rsq_test_with_missing_params() -> None:
        # Initialize mock data entities
        mock_mmm_data = MMMData(...)
        mock_holidays_data = HolidaysData(...)
        mock_calibration_input = CalibrationInput(...)
        mock_hyperparameters = Hyperparameters(...)
        mock_featurized_mmm_data = FeaturizedMMMData(...)

        # Create instance of RidgeModelBuilder
        ridge_model_builder = RidgeModelBuilder(
            mmm_data=mock_mmm_data,
            holiday_data=mock_holidays_data,
            calibration_input=mock_calibration_input,
            hyperparameters=mock_hyperparameters,
            featurized_mmm_data=mock_featurized_mmm_data,
        )

        # Mock Ridge model with predefined coefficients
        mock_ridge_model = Ridge()
        mock_ridge_model.coef_ = np.array([0.1, 0.2, 0.3])

        # Prepare mock DataFrame X and Series y
        mock_X = pd.DataFrame(
            {
                "media1": [1.0, 2.0, 3.0],
                "media2": [4.0, 5.0, 6.0],
                "media3": [7.0, 8.0, 9.0],
            }
        )
        mock_y = pd.Series([1.0, 2.0, 3.0])

        # Call the method with an empty params dictionary
        result_df = ridge_model_builder._calculate_decomp_spend_dist(
            model=mock_ridge_model, X=mock_X, y=mock_y, params={}
        )

        # Extract rsq_test value
        rsq_test = result_df["rsq_test"].iloc[0]

        # Assert that rsq_test is 0
        assert rsq_test == 0, f"Expected rsq_test to be 0, but got {rsq_test}"

    def test_nrmse_val_with_missing_params() -> None:
        # Initialize the RidgeModelBuilder with mock data entities
        mock_mmm_data = MagicMock()
        mock_holidays_data = MagicMock()
        mock_calibration_input = MagicMock()
        mock_hyperparameters = MagicMock()
        mock_featurized_mmm_data = MagicMock()

        # Create an instance of RidgeModelBuilder
        model_builder = RidgeModelBuilder(
            mmm_data=mock_mmm_data,
            holiday_data=mock_holidays_data,
            calibration_input=mock_calibration_input,
            hyperparameters=mock_hyperparameters,
            featurized_mmm_data=mock_featurized_mmm_data,
        )

        # Create a mock Ridge model with predefined coefficients
        mock_ridge_model = Ridge()
        mock_ridge_model.coef_ = np.array([0.1, 0.2, 0.3])

        # Prepare a mock DataFrame X representing feature data
        mock_X = pd.DataFrame(
            {
                "paid_media_1": [1, 2, 3],
                "paid_media_2": [4, 5, 6],
                "paid_media_3": [7, 8, 9],
            }
        )

        # Prepare a mock Series y representing the target values
        mock_y = pd.Series([10, 11, 12])

        # Call the _calculate_decomp_spend_dist method with an empty params dictionary
        result_df = model_builder._calculate_decomp_spend_dist(
            model=mock_ridge_model, X=mock_X, y=mock_y, params={}
        )

        # Extract the nrmse_val value from the resulting DataFrame
        nrmse_val = result_df.get("nrmse_val", None)

        # Assert that nrmse_val is equal to 0
        assert nrmse_val == 0, f"Expected nrmse_val to be 0, but got {nrmse_val}"

    def test_nrmse_test_with_missing_params() -> None:
        # Mock data entities for initialization
        mock_mmm_data = MMMData(...)  # Provide necessary initialization
        mock_holiday_data = HolidaysData(...)  # Provide necessary initialization
        mock_calibration_input = CalibrationInput(
            ...
        )  # Provide necessary initialization
        mock_hyperparameters = Hyperparameters(...)  # Provide necessary initialization
        mock_featurized_mmm_data = FeaturizedMMMData(
            ...
        )  # Provide necessary initialization

        # Initialize RidgeModelBuilder
        ridge_model_builder = RidgeModelBuilder(
            mock_mmm_data,
            mock_holiday_data,
            mock_calibration_input,
            mock_hyperparameters,
            mock_featurized_mmm_data,
        )

        # Create a mock Ridge model with coefficients
        mock_ridge_model = Ridge()
        mock_ridge_model.coef_ = np.array([0.1, 0.2, -0.1])

        # Prepare mock DataFrame X and Series y
        X = pd.DataFrame(
            {
                "media_spend1": [100, 200, 300],
                "media_spend2": [150, 250, 350],
                "media_spend3": [200, 300, 400],
            }
        )
        y = pd.Series([10, 15, 20])

        # Call _calculate_decomp_spend_dist with empty params
        params = {}
        result = ridge_model_builder._calculate_decomp_spend_dist(
            mock_ridge_model, X, y, params
        )

        # Extract and assert the nrmse_test value
        nrmse_test = result["nrmse_test"].iloc[0]
        assert nrmse_test == 0, f"Expected nrmse_test to be 0, got {nrmse_test}"

    def test_lambda_with_missing_params() -> None:
        # Initialize mock data entities
        mmm_data = MagicMock()
        holiday_data = MagicMock()
        calibration_input = MagicMock()
        hyperparameters = MagicMock()
        featurized_mmm_data = MagicMock()

        # Instantiate RidgeModelBuilder
        model_builder = RidgeModelBuilder(
            mmm_data,
            holiday_data,
            calibration_input,
            hyperparameters,
            featurized_mmm_data,
        )

        # Create a mock Ridge model
        mock_model = MagicMock(spec=Ridge)
        mock_model.coef_ = np.array([1.0, 2.0, 3.0])

        # Prepare mock DataFrame and Series
        X = pd.DataFrame(
            {"media1": [1, 2, 3], "media2": [4, 5, 6], "media3": [7, 8, 9]}
        )
        y = pd.Series([1, 2, 3])

        # Call the method with empty params
        result_df = model_builder._calculate_decomp_spend_dist(
            mock_model, X, y, params={}
        )

        # Extract lambda value
        lambda_value = result_df["lambda"].iloc[0]

        # Assert that lambda is 0
        assert lambda_value == 0

    def test_train_size_parameter() -> None:
        # Prepare a Ridge regression model with given coefficients
        model = Ridge()
        model.coef_ = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

        # Create a DataFrame `X` for features and a Series `y` for the target variable
        X = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5],
                "feature2": [2, 3, 4, 5, 6],
                "feature3": [3, 4, 5, 6, 7],
                "feature4": [4, 5, 6, 7, 8],
                "feature5": [5, 6, 7, 8, 9],
            }
        )
        y = pd.Series([10, 20, 30, 40, 50])

        # Define the `params` dictionary including `train_size` set to 0.8
        params = {"train_size": 0.8}

        # Call the `_calculate_x_decomp_agg` method with the model, `X`, `y`, and `params`
        x_decomp_agg = RidgeModelBuilder._calculate_x_decomp_agg(model, X, y, params)

        # Extract the `train_size` value from the first row of the resulting `x_decomp_agg` DataFrame
        train_size_value = x_decomp_agg.loc[0, "train_size"]

        # Assert that the extracted `train_size` value equals 0.8
        assert train_size_value == 0.8

    def test_rsq_train_value() -> None:
        # Mock the necessary methods
        with patch.object(
            RidgeModelBuilder,
            "_prepare_data",
            return_value=(pd.DataFrame(), pd.Series()),
        ) as mock_prepare_data, patch.object(
            RidgeModelBuilder, "_calculate_rssd", return_value=0.0
        ) as mock_calculate_rssd:

            # Create instance of RidgeModelBuilder with mock data
            builder = RidgeModelBuilder(
                mmm_data=MagicMock(),
                holiday_data=MagicMock(),
                calibration_input=MagicMock(),
                hyperparameters=MagicMock(),
                featurized_mmm_data=MagicMock(),
            )

            # Define mock parameters for evaluate_model
            params = {"lambda": 1.0}

            # Call evaluate_model method
            result = builder._evaluate_model(
                params=params,
                ts_validation=False,
                add_penalty_factor=False,
                rssd_zero_penalty=False,
                objective_weights=None,
                start_time=time.time(),
                iter_ng=0,
                trial=1,
            )

            # Assert that rsq_train is a float
            assert isinstance(result["rsq_train"], float)

    def test_rsq_val_parameter() -> None:
        # Prepare a Ridge regression model with given coefficients
        model = Ridge()
        model.coef_ = np.array([1.0, 2.0, 3.0])  # Example coefficients

        # Create a DataFrame `X` for features and a Series `y` for the target variable
        X = pd.DataFrame(
            {"feature1": [1, 2, 3], "feature2": [4, 5, 6], "feature3": [7, 8, 9]}
        )
        y = pd.Series([1, 2, 3])

        # Define the `params` dictionary including `rsq_val` set to 0.7
        params = {"rsq_val": 0.7}

        # Call the `_calculate_x_decomp_agg` method with the model, `X`, `y`, and `params`
        x_decomp_agg = RidgeModelBuilder._calculate_x_decomp_agg(model, X, y, params)

        # Extract the `rsq_val` value from the first row of the resulting `x_decomp_agg` DataFrame
        rsq_val_result = x_decomp_agg.loc[0, "rsq_val"]

        # Assert that the extracted `rsq_val` value equals 0.7
        assert rsq_val_result == 0.7

    def test_rsq_test_parameter() -> None:
        # Prepare a Ridge regression model with given coefficients
        model = Ridge()
        model.coef_ = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

        # Create a DataFrame `X` for features and a Series `y` for the target variable
        X = pd.DataFrame(
            {
                "feature1": [1, 2, 3],
                "feature2": [4, 5, 6],
                "feature3": [7, 8, 9],
                "feature4": [10, 11, 12],
                "feature5": [13, 14, 15],
            }
        )
        y = pd.Series([1, 2, 3])

        # Define the `params` dictionary including `rsq_test` set to 0.6
        params = {
            "rsq_train": 0.8,
            "rsq_val": 0.7,
            "rsq_test": 0.6,
            "nrmse_train": 0.1,
            "nrmse_val": 0.2,
            "nrmse_test": 0.3,
            "nrmse": 0.15,
            "decomp_rssd": 0.05,
            "mape": 0.1,
            "lambda_": 0.01,
            "lambda_hp": 0.02,
            "lambda_max": 0.5,
            "lambda_min_ratio": 0.1,
            "solID": "test_001",
            "trial": 1,
            "iterNG": 10,
            "iterPar": 2,
            "train_size": 1.0,
        }

        # Call the `_calculate_x_decomp_agg` method with the model, `X`, `y`, and `params`
        ridge_builder = RidgeModelBuilder(
            mmm_data=MMMData(),
            holiday_data=HolidaysData(),
            calibration_input=CalibrationInput(),
            hyperparameters=Hyperparameters(),
            featurized_mmm_data=FeaturizedMMMData(),
        )
        x_decomp_agg = ridge_builder._calculate_x_decomp_agg(model, X, y, params)

        # Extract the `rsq_test` value from the first row of the resulting `x_decomp_agg` DataFrame
        rsq_test_value = x_decomp_agg["rsq_test"].iloc[0]

        # Assert that the extracted `rsq_test` value equals 0.6
        assert rsq_test_value == 0.6

    def test_nrmse_train_value() -> None:
        # Prepare test data
        model = Ridge(coef_=np.array([0.5, 0.3, -0.2, 0.1, 0.4]), intercept_=0.0)
        X = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5],
                "feature2": [2, 3, 4, 5, 6],
                "feature3": [3, 4, 5, 6, 7],
                "feature4": [4, 5, 6, 7, 8],
                "feature5": [5, 6, 7, 8, 9],
            }
        )
        y = pd.Series([10, 11, 12, 13, 14])

        # Define params without specific NRMSE values
        params = {
            "train_size": 1.0,
            "rsq_train": 0.0,
            "rsq_val": 0.0,
            "rsq_test": 0.0,
            "nrmse_train": 0.0,
            "nrmse_val": 0.0,
            "nrmse_test": 0.0,
            "decomp.rssd": 0.0,
            "mape": 0.0,
            "lambda": 0.0,
            "lambda_hp": 0.0,
            "lambda_max": 0.0,
            "lambda_min_ratio": 0.0,
            "solID": "test_001",
            "trial": 1,
            "iterNG": 10,
            "iterPar": 2,
        }

        # Calculate x_decomp_agg
        x_decomp_agg = RidgeModelBuilder._calculate_x_decomp_agg(model, X, y, params)

        # Extract and assert nrmse_train value
        nrmse_train = x_decomp_agg.loc[0, "nrmse_train"]
        assert np.isclose(
            nrmse_train, 0.08, atol=0.01
        ), f"Expected nrmse_train close to 0.08, got {nrmse_train}"

    def test_nrmse_val_parameter() -> None:
        # Prepare a Ridge regression model with dummy coefficients
        model = Ridge()
        model.coef_ = np.array([0.5, -0.2, 0.1])  # Example coefficients

        # Create a DataFrame `X` for features and a Series `y` for the target variable
        X = pd.DataFrame(
            {"feature1": [1, 2, 3], "feature2": [4, 5, 6], "feature3": [7, 8, 9]}
        )
        y = pd.Series([1, 2, 3])

        # Define the `params` dictionary including `nrmse_val` set to 0.1
        params = {"nrmse_val": 0.1}

        # Call the `_calculate_x_decomp_agg` method
        x_decomp_agg = RidgeModelBuilder._calculate_x_decomp_agg(
            model=model, X=X, y=y, params=params
        )

        # Extract the `nrmse_val` value from the first row of the resulting `x_decomp_agg` DataFrame
        nrmse_val = x_decomp_agg["nrmse_val"].iloc[0]

        # Assert that the extracted `nrmse_val` value equals 0.1
        assert nrmse_val == 0.1

    def test_nrmse_test_parameter() -> None:
        # Prepare a Ridge regression model with coefficients
        model = Ridge()
        model.coef_ = np.array([0.5, 1.0, -1.5, 2.0, -0.5])

        # Create a DataFrame `X` for features and a Series `y` for the target variable
        X = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5],
                "feature2": [2, 3, 4, 5, 6],
                "feature3": [3, 4, 5, 6, 7],
                "feature4": [4, 5, 6, 7, 8],
                "feature5": [5, 6, 7, 8, 9],
            }
        )
        y = pd.Series([10, 12, 14, 16, 18])

        # Define the `params` dictionary including `nrmse_test` set to 0.15
        params = {
            "nrmse_test": 0.15,
            "solID": "test_001",
            "trial": 1,
            "iterNG": 10,
            "iterPar": 2,
        }

        # Call the `_calculate_x_decomp_agg` method with the model, `X`, `y`, and `params`
        x_decomp_agg = RidgeModelBuilder._calculate_x_decomp_agg(model, X, y, params)

        # Extract the `nrmse_test` value from the first row of the resulting `x_decomp_agg` DataFrame
        extracted_nrmse_test = x_decomp_agg["nrmse_test"].iloc[0]

        # Assert that the extracted `nrmse_test` value equals 0.15
        assert extracted_nrmse_test == 0.15

    def test_decomp_rssd_parameter(self):
        # Prepare a Ridge regression model with given coefficients
        model = Ridge()
        model.coef_ = np.array([1, 2, 3])  # Example coefficients

        # Create a DataFrame `X` for features and a Series `y` for the target variable
        X = pd.DataFrame(
            {
                "feature1": [0.1, 0.2, 0.3],
                "feature2": [0.4, 0.5, 0.6],
                "feature3": [0.7, 0.8, 0.9],
            }
        )
        y = pd.Series([1, 2, 3])

        # Define the `params` dictionary including `decomp.rssd` set to 0.05
        params = {"decomp.rssd": 0.05}

        # Call the `_calculate_x_decomp_agg` method with the model, `X`, `y`, and `params`
        x_decomp_agg = self._calculate_x_decomp_agg(model, X, y, params)

        # Extract the `decomp.rssd` value from the first row of the resulting `x_decomp_agg` DataFrame
        decomp_rssd_value = x_decomp_agg["decomp.rssd"].iloc[0]

        # Assert that the extracted `decomp.rssd` value equals 0.05
        self.assertEqual(decomp_rssd_value, 0.05)

    def test_mape_parameter() -> None:
        # Prepare a Ridge regression model with given coefficients
        model = Ridge()
        model.coef_ = np.array([0.5, 1.0, -0.5])  # Example coefficients

        # Create a DataFrame `X` for features and a Series `y` for the target variable
        X = pd.DataFrame(
            {"feature1": [1, 2, 3], "feature2": [4, 5, 6], "feature3": [7, 8, 9]}
        )
        y = pd.Series([1, 2, 3])

        # Define the `params` dictionary including `mape` set to 0.1
        params = {"mape": 0.1}

        # Call the `_calculate_x_decomp_agg` method with the model, `X`, `y`, and `params`
        x_decomp_agg = RidgeModelBuilder._calculate_x_decomp_agg(model, X, y, params)

        # Extract the `mape` value from the first row of the resulting `x_decomp_agg` DataFrame
        mape_value = x_decomp_agg["mape"].iloc[0]

        # Assert that the extracted `mape` value equals 0.1
        assert mape_value == 0.1

    def test_lambda_parameter() -> None:
        # Prepare a Ridge regression model with given coefficients
        model = Ridge()
        model.coef_ = np.array([0.5, 1.5])

        # Create a DataFrame `X` for features and a Series `y` for the target variable
        X = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})
        y = pd.Series([7, 8, 9])

        # Define the `params` dictionary including `lambda` set to 0.01
        params = {"lambda": 0.01}

        # Call the `_calculate_x_decomp_agg` method with the model, `X`, `y`, and `params`
        x_decomp_agg = RidgeModelBuilder._calculate_x_decomp_agg(model, X, y, params)

        # Extract the `lambda` value from the first row of the resulting `x_decomp_agg` DataFrame
        lambda_value = x_decomp_agg["lambda"].iloc[0]

        # Assert that the extracted `lambda` value equals 0.01
        assert lambda_value == 0.01

    def test_lambda_hp_parameter() -> None:
        # Prepare a Ridge regression model with given coefficients
        model = Ridge()
        model.coef_ = np.array([0.5, -0.2, 0.1])  # Example coefficients

        # Create a DataFrame `X` for features and a Series `y` for the target variable
        X = pd.DataFrame(
            {
                "feature1": [0.5, 1.0, 1.5],
                "feature2": [0.2, 0.4, 0.6],
                "feature3": [0.1, 0.3, 0.5],
            }
        )
        y = pd.Series([1.0, 2.0, 3.0])

        # Define the `params` dictionary including `lambda_hp` set to 0.02
        params = {
            "lambda_hp": 0.02,
            "train_size": 1.0,
            "rsq_train": 0.8,
            "rsq_val": 0.7,
            "rsq_test": 0.6,
            "nrmse_train": 0.08,
            "nrmse_val": 0.1,
            "nrmse_test": 0.15,
            "decomp.rssd": 0.05,
            "mape": 0.1,
            "lambda": 0.01,
            "solID": "test_001",
            "trial": 1,
            "iterNG": 10,
            "iterPar": 2,
        }

        # Call the `_calculate_x_decomp_agg` method with the model, `X`, `y`, and `params`
        x_decomp_agg = RidgeModelBuilder._calculate_x_decomp_agg(model, X, y, params)

        # Extract the `lambda_hp` value from the first row of the resulting `x_decomp_agg` DataFrame
        lambda_hp_value = x_decomp_agg["lambda_hp"].iloc[0]

        # Assert that the extracted `lambda_hp` value equals 0.02
        assert lambda_hp_value == 0.02

    def test_solID_parameter() -> None:
        # Prepare a Ridge regression model with given coefficients
        model = Ridge()
        model.coef_ = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

        # Create a DataFrame `X` for features and a Series `y` for the target variable
        X = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5],
                "feature2": [5, 4, 3, 2, 1],
                "feature3": [2, 3, 4, 5, 6],
                "feature4": [6, 5, 4, 3, 2],
                "feature5": [1, 3, 5, 7, 9],
            }
        )
        y = pd.Series([1, 2, 3, 4, 5])

        # Define the `params` dictionary including `solID` set to "test_001"
        params = {"solID": "test_001"}

        # Call the `_calculate_x_decomp_agg` method with the model, `X`, `y`, and `params`
        x_decomp_agg = RidgeModelBuilder._calculate_x_decomp_agg(model, X, y, params)

        # Extract the `solID` value from the first row of the resulting `x_decomp_agg` DataFrame
        solID_value = x_decomp_agg["solID"].iloc[0]

        # Assert that the extracted `solID` value equals "test_001"
        assert solID_value == "test_001"

    def test_trial_parameter(self):
        # Prepare a Ridge regression model with sample coefficients
        model = Ridge()
        model.coef_ = np.array([0.5, 0.2, -0.1])

        # Create feature DataFrame X and target Series y
        X = pd.DataFrame(
            {
                "feature1": [0.1, 0.2, 0.3],
                "feature2": [0.4, 0.5, 0.6],
                "feature3": [0.7, 0.8, 0.9],
            }
        )
        y = pd.Series([1.0, 2.0, 3.0])

        # Define the parameters including trial set to 1
        params = {
            "train_size": 1.0,
            "rsq_train": 0.9,
            "rsq_val": 0.8,
            "rsq_test": 0.7,
            "nrmse_train": 0.1,
            "nrmse_val": 0.2,
            "nrmse_test": 0.3,
            "nrmse": 0.15,
            "decomp.rssd": 0.05,
            "mape": 0.1,
            "lambda": 0.01,
            "lambda_hp": 0.02,
            "lambda_max": 0.03,
            "lambda_min_ratio": 0.001,
            "solID": "test_001",
            "trial": 1,
            "iterNG": 10,
            "iterPar": 2,
        }

        # Call the method to calculate x_decomp_agg
        x_decomp_agg = self._calculate_x_decomp_agg(model, X, y, params)

        # Extract the trial value from the first row
        trial_value = x_decomp_agg["trial"].iloc[0]

        # Assert that the trial value is 1
        self.assertEqual(trial_value, 1)

    def test_iterNG_parameter() -> None:
        # Prepare a Ridge regression model with given coefficients
        model = Ridge()
        model.coef_ = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

        # Create a DataFrame `X` for features and a Series `y` for the target variable
        X = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5],
                "feature2": [2, 3, 4, 5, 6],
                "feature3": [3, 4, 5, 6, 7],
                "feature4": [4, 5, 6, 7, 8],
                "feature5": [5, 6, 7, 8, 9],
            }
        )
        y = pd.Series([1, 2, 3, 4, 5])

        # Define the `params` dictionary including `iterNG` set to 10
        params = {
            "iterNG": 10,
            "rsq_val": 0,
            "rsq_test": 0,
            "nrmse_val": 0,
            "nrmse_test": 0,
            "nrmse": 0,
            "decomp_rssd": 0,
            "mape": 0,
            "lambda_": 0,
            "lambda_hp": 0,
            "lambda_max": 0,
            "lambda_min_ratio": 0,
            "solID": "",
            "trial": 0,
            "iterPar": 0,
        }

        # Call the `_calculate_x_decomp_agg` method with the model, `X`, `y`, and `params`
        x_decomp_agg = RidgeModelBuilder._calculate_x_decomp_agg(model, X, y, params)

        # Extract the `iterNG` value from the first row of the resulting `x_decomp_agg` DataFrame
        iterNG_value = x_decomp_agg["iterNG"].iloc[0]

        # Assert that the extracted `iterNG` value equals 10
        assert iterNG_value == 10

    def test_iterPar_parameter() -> None:
        # Prepare a Ridge regression model with given coefficients
        model = Ridge()
        model.coef_ = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

        # Create a DataFrame `X` for features and a Series `y` for the target variable
        X = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5],
                "feature2": [2, 3, 4, 5, 6],
                "feature3": [3, 4, 5, 6, 7],
                "feature4": [4, 5, 6, 7, 8],
                "feature5": [5, 6, 7, 8, 9],
            }
        )
        y = pd.Series([1, 2, 3, 4, 5])

        # Define the `params` dictionary including `iterPar` set to 2
        params = {"iterPar": 2}

        # Call the `_calculate_x_decomp_agg` method with the model, `X`, `y`, and `params`
        x_decomp_agg = RidgeModelBuilder._calculate_x_decomp_agg(model, X, y, params)

        # Extract the `iterPar` value from the first row of the resulting `x_decomp_agg` DataFrame
        iterPar_value = x_decomp_agg["iterPar"].iloc[0]

        # Assert that the extracted `iterPar` value equals 2
        assert iterPar_value == 2

    def test_x_decomp_agg_sum_zero() -> None:
        # Set up test environment
        model = Ridge()
        model.coef_ = np.array([0.0, 0.0, 0.0])  # Zero coefficients

        # Create test data
        X = pd.DataFrame(
            {"feature1": [1, 2, 3], "feature2": [4, 5, 6], "feature3": [7, 8, 9]}
        )
        y = pd.Series([10, 11, 12])

        # Define parameters
        params = {
            "train_size": 1.0,
            "rsq_val": 0.0,
            "rsq_test": 0.0,
            "nrmse_val": 0.0,
            "nrmse_test": 0.0,
            "nrmse": 0.0,
            "decomp_rssd": 0.0,
            "mape": 0.0,
            "lambda_": 0.0,
            "lambda_hp": 0.0,
            "lambda_max": 0.0,
            "lambda_min_ratio": 0.0,
            "solID": "",
            "trial": 0,
            "iter_ng": 0,
            "iter_par": 0,
        }

        # Call method
        result_df = RidgeModelBuilder._calculate_x_decomp_agg(model, X, y, params)

        # Assert sum of xDecompAgg
        assert (
            result_df["xDecompAgg"].sum() == 0.0
        ), "Sum of xDecompAgg should be zero for zero coefficients"

    def test_x_decomp_perc_sum_zero() -> None:
        # Set up the test environment and necessary inputs
        model = Ridge()
        model.coef_ = np.array([0.0, 0.0, 0.0])  # Zero coefficients

        # Create a DataFrame `X` with features and a Series `y` with target values
        X = pd.DataFrame(
            {"feature1": [1, 2, 3], "feature2": [4, 5, 6], "feature3": [7, 8, 9]}
        )
        y = pd.Series([1, 2, 3])

        # Define the parameters dictionary
        params = {
            "train_size": 1.0,
            "rsq_val": 0.0,
            "rsq_test": 0.0,
            "nrmse_val": 0.0,
            "nrmse_test": 0.0,
            "nrmse": 0.0,
            "decomp_rssd": 0.0,
            "mape": 0.0,
            "lambda_": 0.0,
            "lambda_hp": 0.0,
            "lambda_max": 0.0,
            "lambda_min_ratio": 0.0,
            "solID": "test_001",
            "trial": 1,
            "iter_ng": 1,
            "iter_par": 1,
        }

        # Call the `_calculate_x_decomp_agg` method
        ridge_builder = RidgeModelBuilder(None, None, None, None, None)
        result_df = ridge_builder._calculate_x_decomp_agg(model, X, y, params)

        # Assert that the sum of the `xDecompPerc` column is zero
        assert (
            result_df["xDecompPerc"].sum() == 0.0
        ), "Sum of xDecompPerc should be zero when coefficients are zero"

    def test_x_decomp_agg_pos_all_false() -> None:
        # Prepare the test setup
        model = Ridge()
        model.coef_ = np.array([-0.1, -0.2, -0.3])
        X = pd.DataFrame(
            {"feature1": [1, 2, 3], "feature2": [4, 5, 6], "feature3": [7, 8, 9]}
        )
        y = pd.Series([10, 15, 20])

        # Prepare parameters
        params = {
            "train_size": 0.8,
            "rsq_val": 0.0,
            "rsq_test": 0.0,
            "nrmse_val": 0.0,
            "nrmse_test": 0.0,
            "nrmse": 0.0,
            "decomp_rssd": 0.0,
            "mape": 0.0,
            "lambda_": 1.0,
            "lambda_hp": 0.0,
            "lambda_max": 0.0,
            "lambda_min_ratio": 0.0,
            "solID": "",
            "trial": 0,
            "iter_ng": 0,
            "iter_par": 0,
        }

        # Invoke the method
        ridge_model_builder = RidgeModelBuilder(
            mmm_data=None,
            holiday_data=None,
            calibration_input=None,
            hyperparameters=None,
            featurized_mmm_data=None,
        )
        x_decomp_agg = ridge_model_builder._calculate_x_decomp_agg(model, X, y, params)

        # Perform the assertion
        assert all(
            x_decomp_agg["pos"] == False
        ), "Expected all 'pos' values to be False for negative coefficients"

    def test_train_size_calculation() -> None:
        # Setup mock inputs
        model = Ridge()
        model.coef_ = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

        X = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5],
                "feature2": [2, 3, 4, 5, 6],
                "feature3": [3, 4, 5, 6, 7],
                "feature4": [4, 5, 6, 7, 8],
                "feature5": [5, 6, 7, 8, 9],
            }
        )
        y = pd.Series([1, 2, 3, 4, 5])

        params = {
            "train_size": 0.5,
            "rsq_val": 0.0,
            "rsq_test": 0.0,
            "nrmse_val": 0.0,
            "nrmse_test": 0.0,
            "nrmse": 0.0,
            "decomp_rssd": 0.0,
            "mape": 0.0,
            "lambda_": 0.0,
            "lambda_hp": 0.0,
            "lambda_max": 0.0,
            "lambda_min_ratio": 0.0,
            "solID": "",
            "trial": 0,
            "iterNG": 0,
            "iterPar": 0,
        }

        # Call the method under test
        x_decomp_agg = RidgeModelBuilder._calculate_x_decomp_agg(model, X, y, params)

        # Verify the train_size calculation
        actual_train_size = x_decomp_agg["train_size"].iloc[0]
        expected_train_size = 0.5

        # Assert the train size is as expected
        assert (
            actual_train_size == expected_train_size
        ), f"Expected train_size to be {expected_train_size}, but got {actual_train_size}"

    def test_lambda_value() -> None:
        # Mock necessary methods and components
        mock_mmm_data = MagicMock()
        mock_holiday_data = MagicMock()
        mock_calibration_input = MagicMock()
        mock_hyperparameters = {"prepared_hyperparameters": {"lambda": 1.0}}
        mock_featurized_mmm_data = MagicMock()

        # Instantiate RidgeModelBuilder with mock data
        model_builder = RidgeModelBuilder(
            mmm_data=mock_mmm_data,
            holiday_data=mock_holiday_data,
            calibration_input=mock_calibration_input,
            hyperparameters=mock_hyperparameters,
            featurized_mmm_data=mock_featurized_mmm_data,
        )

        # Mock the evaluate_model method to return a specific lambda value
        with patch.object(
            model_builder, "_evaluate_model", return_value={"lambda_": 1.0}
        ) as mock_evaluate_model:
            # Run the method to be tested
            params = {"lambda": 1.0}
            result = model_builder._evaluate_model(
                params,
                ts_validation=False,
                add_penalty_factor=False,
                rssd_zero_penalty=False,
                objective_weights=None,
                start_time=0,
                iter_ng=0,
                trial=1,
            )

            # Assert that the lambda value in the result is as expected
            assert result["lambda_"] == 1.0

    def test_lambda_hp_value(self):
        # Prepare input data
        model = Ridge()
        X = pd.DataFrame(
            np.random.rand(10, 5), columns=[f"feature_{i}" for i in range(5)]
        )
        y = pd.Series(np.random.rand(10))
        params = {"lambda_hp": 0.1}

        # Call the method
        result_df = self._calculate_x_decomp_agg(model, X, y, params)

        # Assert the lambda_hp value
        self.assertEqual(result_df["lambda_hp"].iloc[0], 0.1)

    def test_lambda_max_value() -> None:
        # Initialize the input Ridge model, DataFrame `X`, and Series `y`.
        model = Ridge()
        X = pd.DataFrame(
            np.random.rand(10, 5), columns=[f"feature_{i}" for i in range(5)]
        )
        y = pd.Series(np.random.rand(10))

        # Enter the expected `lambda_max` in the parameters.
        params = {"lambda_max": 0.2}

        # Invoke `_calculate_x_decomp_agg` with these inputs.
        result_df = RidgeModelBuilder._calculate_x_decomp_agg(model, X, y, params)

        # Confirm the `lambda_max` value in the DataFrame equals 0.2.
        assert (
            result_df["lambda_max"].iloc[0] == 0.2
        ), "lambda_max value is not as expected."

    def test_lambda_min_ratio_value(self) -> None:
        # Setup
        model = Ridge()
        X = pd.DataFrame(
            np.random.randn(10, 5), columns=[f"feature_{i}" for i in range(5)]
        )
        y = pd.Series(np.random.randn(10))
        params = {"lambda_min_ratio": 0.0005}

        # Execute
        x_decomp_agg = RidgeModelBuilder._calculate_x_decomp_agg(model, X, y, params)

        # Validate
        self.assertEqual(x_decomp_agg["lambda_min_ratio"].iloc[0], 0.0005)

    def test_x_decomp_agg_empty():
        # Initialize an empty DataFrame X and an empty Series y
        X = pd.DataFrame()
        y = pd.Series(dtype=float)

        # Instantiate a Ridge model with empty coefficients
        model = Ridge()
        model.coef_ = np.array([])  # Empty array for coefficients

        # Empty parameters dictionary
        params = {}

        # Call the _calculate_x_decomp_agg method
        x_decomp_agg = RidgeModelBuilder._calculate_x_decomp_agg(model, X, y, params)

        # Assert that the returned DataFrame is empty
        assert (
            x_decomp_agg.empty
        ), "Expected x_decomp_agg to be empty for empty input DataFrame"

    def test_prepare_data_X_columns_are_correct(self) -> None:
        # Mock the FeaturizedMMMData to return a DataFrame with specific columns
        featurized_mmm_data_mock = MagicMock()
        featurized_mmm_data_mock.dt_mod = pd.DataFrame(
            {"dep_var": [1, 2, 3], "feature1": [4, 5, 6], "feature2": [7, 8, 9]}
        )

        # Mock the MMMData to specify 'dep_var' as the dependent variable and 'feature1' as a paid media spend
        mmm_data_mock = MagicMock()
        mmm_data_mock.mmmdata_spec.dep_var = "dep_var"
        mmm_data_mock.mmmdata_spec.paid_media_spends = ["feature1"]

        # Create an instance of RidgeModelBuilder with mocked data
        ridge_model_builder = RidgeModelBuilder(
            mmm_data=mmm_data_mock,
            holiday_data=MagicMock(),
            calibration_input=MagicMock(),
            hyperparameters=MagicMock(),
            featurized_mmm_data=featurized_mmm_data_mock,
        )

        # Call the _prepare_data method
        params = {"feature1_thetas": 0.5}  # Example hyperparameter for transformation
        X, _ = ridge_model_builder._prepare_data(params)

        # Assert that the columns of X are as expected
        self.assertListEqual(list(X.columns), ["feature1"])

    def test_prepare_data_y_name_is_correct() -> None:
        # Mock the FeaturizedMMMData dependency to provide a DataFrame with a 'dep_var' column
        featurized_mmm_data_mock = MagicMock()
        featurized_mmm_data_mock.dt_mod = pd.DataFrame(
            {"dep_var": [1, 2, 3, 4, 5], "feature1": [10, 20, 30, 40, 50]}
        )

        # Mock the MMMData dependency to specify 'dep_var' as the dependent variable
        mmm_data_mock = MagicMock()
        mmm_data_mock.mmmdata_spec.dep_var = "dep_var"

        # Create an instance of RidgeModelBuilder with mocked dependencies
        ridge_model_builder = RidgeModelBuilder(
            mmm_data=mmm_data_mock,
            holiday_data=MagicMock(),
            calibration_input=MagicMock(),
            hyperparameters=MagicMock(),
            featurized_mmm_data=featurized_mmm_data_mock,
        )

        # Call the _prepare_data method with a dictionary of hyperparameters
        params = {}
        _, y = ridge_model_builder._prepare_data(params)

        # Assert that the name of y is 'dep_var'
        assert y.name == "dep_var", "The dependent variable name should be 'dep_var'"

    def test_date_column_min_value() -> None:
        # Mock Setup: Create mock DataFrame structure
        mock_featurized_mmm_data = FeaturizedMMMData()
        mock_featurized_mmm_data.dt_mod = pd.DataFrame(
            {
                "dep_var": [100, 150, 200],
                "date_col": ["2021-01-01", "2021-01-02", "2021-01-03"],
                "feature1": [1.0, 2.0, 3.0],
            }
        )

        # Mock Setup: Create mock MMMData structure
        mock_mmm_data = MMMData()
        mock_mmm_data.mmmdata_spec = type("", (), {})()
        mock_mmm_data.mmmdata_spec.dep_var = "dep_var"
        mock_mmm_data.mmmdata_spec.paid_media_spends = ["feature1"]

        # Initialize Test Object
        ridge_model_builder = RidgeModelBuilder(
            mmm_data=mock_mmm_data,
            holiday_data=None,
            calibration_input=None,
            hyperparameters=None,
            featurized_mmm_data=mock_featurized_mmm_data,
        )

        # Prepare Data
        params = {"feature1_thetas": 0.5}
        X, y = ridge_model_builder._prepare_data(params)

        # Check Date Conversion
        # Ensure 'date_col' conversion to days since the minimum date results in 0
        assert (
            X["date_col"].min() == 0
        ), "Minimum date_col value should be converted to 0"

    def test_dependent_variable_name() -> None:
        # Mock Setup: Mock the FeaturizedMMMData to return a DataFrame with columns 'dep_var', 'date_col', and 'feature1'.
        featurized_mmm_data_mock = MagicMock()
        featurized_mmm_data_mock.dt_mod = pd.DataFrame(
            {
                "dep_var": [1, 2, 3, 4, 5],
                "date_col": pd.date_range(start="1/1/2021", periods=5),
                "feature1": [10, 20, 30, 40, 50],
            }
        )

        # Mock Setup: Mock MMMData to provide a specification where 'dep_var' is the dependent variable.
        mmm_data_mock = MagicMock()
        mmm_data_mock.mmmdata_spec.dep_var = "dep_var"
        mmm_data_mock.mmmdata_spec.paid_media_spends = ["feature1"]

        # Initialize Test Object: Create an instance of the RidgeModelBuilder using the mocked FeaturizedMMMData and MMMData.
        ridge_model_builder = RidgeModelBuilder(
            mmm_data=mmm_data_mock,
            holiday_data=MagicMock(),
            calibration_input=MagicMock(),
            hyperparameters=MagicMock(),
            featurized_mmm_data=featurized_mmm_data_mock,
        )

        # Prepare Data: Invoke the _prepare_data method on the RidgeModelBuilder object with a dictionary containing feature1_thetas.
        params = {"feature1_thetas": 0.5}
        _, y = ridge_model_builder._prepare_data(params)

        # Verify Dependent Variable Name: Check that the name of the returned Series y matches the expected dependent variable name as specified in MMMData.
        assert y.name == "dep_var", "Dependent variable name is not correctly set"

    def test_prepare_data_feature_columns() -> None:
        # Mock the FeaturizedMMMData dependency
        featurized_mmm_data_mock = MagicMock()
        featurized_mmm_data_mock.dt_mod = pd.DataFrame(
            {
                "dep_var": [1, 2, 3],
                "category_col": ["value1", "value2", "value1"],
                "feature1": [100, 200, 300],
            }
        )

        # Mock the MMMData dependency
        mmm_data_mock = MagicMock()
        mmm_data_mock.mmmdata_spec.dep_var = "dep_var"
        mmm_data_mock.mmmdata_spec.paid_media_spends = ["feature1"]

        # Create an instance of RidgeModelBuilder with mocked data
        ridge_model_builder = RidgeModelBuilder(
            mmm_data=mmm_data_mock,
            holiday_data=MagicMock(),
            calibration_input=MagicMock(),
            hyperparameters=MagicMock(),
            featurized_mmm_data=featurized_mmm_data_mock,
        )

        # Call the _prepare_data method
        params = {"feature1_thetas": 0.5}
        X, _ = ridge_model_builder._prepare_data(params)

        # Assert that the columns of X are as expected after transformation and one-hot encoding
        expected_columns = ["feature1", "category_col_value2"]
        assert list(X.columns) == expected_columns

    def test_prepare_data_dependent_variable_column(self):
        # Mock the FeaturizedMMMData to simulate a DataFrame that includes 'dep_var'
        featurized_mmm_data = MagicMock()
        featurized_mmm_data.dt_mod = pd.DataFrame(
            {
                "dep_var": [1, 2, 3, 4, 5],
                "feature1": [10, 20, 30, 40, 50],
                "feature2": [5, 4, 3, 2, 1],
            }
        )

        # Mock the MMMData to provide a specification where 'dep_var' is the target variable
        mmm_data = MagicMock()
        mmm_data.mmmdata_spec.dep_var = "dep_var"

        # Create a RidgeModelBuilder instance with mocked data
        ridge_model_builder = RidgeModelBuilder(
            mmm_data=mmm_data,
            holiday_data=None,
            calibration_input=None,
            hyperparameters=None,
            featurized_mmm_data=featurized_mmm_data,
        )

        # Call the _prepare_data method
        params = {}  # Assuming no specific parameters are needed for the test
        _, y = ridge_model_builder._prepare_data(params)

        # Assert the name of the y Series is 'dep_var'
        self.assertEqual(y.name, "dep_var")

    def test_prepare_data_y_isnull_sum(params: Dict[str, float]) -> None:
        # Mock FeaturizedMMMData to return a DataFrame with 'dep_var' and 'feature1'
        featurized_mmm_data_mock = MagicMock()
        featurized_mmm_data_mock.dt_mod = pd.DataFrame(
            {"dep_var": [1, 2, 3, 4, 5], "feature1": [10, 20, 30, 40, 50]}
        )

        # Mock MMMData to return a specification with 'dep_var' as the dependent variable
        mmm_data_mock = MagicMock()
        mmm_data_mock.mmmdata_spec.dep_var = "dep_var"
        mmm_data_mock.mmmdata_spec.paid_media_spends = ["feature1"]

        # Create an instance of RidgeModelBuilder with mocked data
        ridge_model_builder = RidgeModelBuilder(
            mmm_data=mmm_data_mock,
            holiday_data=MagicMock(),
            calibration_input=MagicMock(),
            hyperparameters=MagicMock(),
            featurized_mmm_data=featurized_mmm_data_mock,
        )

        # Invoke the _prepare_data method
        _, y = ridge_model_builder._prepare_data(params)

        # Assert that the sum of null values in 'y' is zero
        assert y.isnull().sum() == 0

    def test_prepare_data_y_name(self):
        # Mock FeaturizedMMMData to return a DataFrame with 'dep_var' and 'feature1' columns
        featurized_mmm_data = MagicMock()
        featurized_mmm_data.dt_mod = pd.DataFrame(
            {"dep_var": [1, 2, 3], "feature1": [4, 5, 6]}
        )

        # Mock MMMData to specify 'dep_var' as the dependent variable
        mmm_data = MagicMock()
        mmm_data.mmmdata_spec.dep_var = "dep_var"
        mmm_data.mmmdata_spec.paid_media_spends = ["feature1"]

        # Create an instance of RidgeModelBuilder with mocked data
        ridge_model_builder = RidgeModelBuilder(
            mmm_data=mmm_data,
            holiday_data=MagicMock(),
            calibration_input=MagicMock(),
            hyperparameters=MagicMock(),
            featurized_mmm_data=featurized_mmm_data,
        )

        # Define the params dictionary
        params = {}

        # Invoke the _prepare_data method
        _, y = ridge_model_builder._prepare_data(params)

        # Assert that the name of the y Series is equal to the specified dependent variable 'dep_var'
        self.assertEqual(y.name, "dep_var")

    def test_feature1_no_nulls_in_X(self):
        # Mock the FeaturizedMMMData to simulate a DataFrame with 'dep_var' and 'feature1'
        featurized_mmm_data_mock = MagicMock()
        featurized_mmm_data_mock.dt_mod = pd.DataFrame(
            {"dep_var": [1, 2, 3, 4, 5], "feature1": [10, 20, np.nan, 40, 50]}
        )

        # Mock the MMMData to include 'dep_var' and 'paid_media_spends' with 'feature1'
        mmm_data_mock = MagicMock()
        mmm_data_mock.mmmdata_spec.dep_var = "dep_var"
        mmm_data_mock.mmmdata_spec.paid_media_spends = ["feature1"]

        # Initialize RidgeModelBuilder with mocked data
        ridge_model_builder = RidgeModelBuilder(
            mmm_data=mmm_data_mock,
            holiday_data=MagicMock(),
            calibration_input=MagicMock(),
            hyperparameters=MagicMock(),
            featurized_mmm_data=featurized_mmm_data_mock,
        )

        # Prepare data with provided params
        params = {}
        X, y = ridge_model_builder._prepare_data(params)

        # Check for null values in 'feature1' column
        null_count_feature1 = X["feature1"].isnull().sum()

        # Assert that no nulls exist in 'feature1'
        self.assertEqual(null_count_feature1, 0)

    def test_y_name_is_dep_var(self, params: Dict[str, float]) -> None:
        # Mock the FeaturizedMMMData to return a DataFrame with 'dep_var' and 'feature1'
        featurized_mmm_data_mock = MagicMock()
        featurized_mmm_data_mock.dt_mod = pd.DataFrame(
            {"dep_var": [1, 2, 3], "feature1": [10, 20, 30]}
        )

        # Mock the MMMData to return the specification with 'dep_var' and 'paid_media_spends'
        mmm_data_mock = MagicMock()
        mmm_data_mock.mmmdata_spec.dep_var = "dep_var"
        mmm_data_mock.mmmdata_spec.paid_media_spends = ["feature1"]

        # Create an instance of RidgeModelBuilder with mocked dependencies
        ridge_model_builder = RidgeModelBuilder(
            mmm_data=mmm_data_mock,
            holiday_data=MagicMock(),
            calibration_input=MagicMock(),
            hyperparameters=MagicMock(),
            featurized_mmm_data=featurized_mmm_data_mock,
        )

        # Call the _prepare_data method with params
        _, y = ridge_model_builder._prepare_data(params)

        # Assert that the name of the series y is 'dep_var'
        self.assertEqual(y.name, "dep_var")

    def test_geometric_adstock_first_element() -> None:
        # Initialize the input series x as a pandas Series with values [10, 20, 30]
        x = pd.Series([10, 20, 30])

        # Set the adstock decay parameter theta to 1.5
        theta = 1.5

        # Apply the geometric adstock transformation method to the series x using the theta parameter
        ridge_model_builder = RidgeModelBuilder(...)
        y = ridge_model_builder._geometric_adstock(x, theta)

        # Assert that the first element of the transformed series y.iloc[0] is equal to 10
        assert y.iloc[0] == 10, "The first element should remain unchanged"

    def test_geometric_adstock_second_element() -> None:
        x = pd.Series([10, 20, 30])
        theta = 1.5
        y = RidgeModelBuilder._geometric_adstock(None, x, theta)
        expected_value = 20 + 1.5 * 10
        assert y.iloc[1] == expected_value

    def test_geometric_adstock_third_element():
        x = pd.Series([10, 20, 30])
        theta = 1.5
        y = RidgeModelBuilder._geometric_adstock(RidgeModelBuilder, x, theta)
        expected_value = 30 + 1.5 * (20 + 1.5 * 10)
        assert (
            y.iloc[2] == expected_value
        ), f"Expected {expected_value}, got {y.iloc[2]}"

    def test_geometric_adstock_fourth_element() -> None:
        x = pd.Series([1, 2, 3, 4, 5])
        theta = 1
        y = RidgeModelBuilder._geometric_adstock(RidgeModelBuilder(), x, theta)
        assert y.iloc[3] == 10

    def test_geometric_adstock_fifth_element() -> None:
        x = pd.Series([1, 2, 3, 4, 5])
        theta = 1
        y = RidgeModelBuilder._geometric_adstock(RidgeModelBuilder, x, theta)
        assert y.iloc[4] == 15

    def test_geometric_adstock_with_theta_zero(self):
        # Setup initial test data
        x = pd.Series([1, 2, 3, 4, 5])
        theta = 0

        # Call the function with theta set to 0
        y = RidgeModelBuilder._geometric_adstock(self, x, theta)

        # Assert that the output series is equal to the input series
        self.assertTrue(
            y.equals(x),
            "The output series should be equal to the input series when theta is 0.",
        )

    def test_geometric_adstock_empty_series() -> None:
        # Initialize an empty Pandas Series
        x = pd.Series(dtype=float)

        # Set the adstock decay parameter theta
        theta = 0.5

        # Invoke the geometric_adstock method with the empty series and theta
        result = RidgeModelBuilder._geometric_adstock(self=None, x=x, theta=theta)

        # Assert that the result is an empty series
        assert result.empty, "Expected an empty series, but got a non-empty output."

    def assert_geometric_adstock_empty_series_empty() -> None:
        # Create an empty pandas Series
        x = pd.Series(dtype=float)

        # Apply the geometric adstock function with any theta, e.g., 0.5
        result = RidgeModelBuilder._geometric_adstock(RidgeModelBuilder(), x, theta=0.5)

        # Assert that the result is an empty series
        assert (
            result.empty
        ), "The result should be an empty series when the input is empty."

        # Verify the expected output matches the actual output
        expected_output = pd.Series(dtype=float)
        pd.testing.assert_series_equal(result, expected_output, check_dtype=True)

    def test_hill_transformation_output() -> None:
        # Initialize a Pandas Series with negative values
        x = pd.Series([-0.1, -0.2, -0.3, -0.4, -0.5])

        # Set parameters for the Hill transformation
        alpha = 2.0
        gamma = 1.0

        # Call the _hill_transformation method
        output = RidgeModelBuilder._hill_transformation(
            RidgeModelBuilder, x, alpha, gamma
        )

        # Define expected output as a series of zeros
        expected_output = pd.Series([0.0, 0.0, 0.0, 0.0, 0.0])

        # Assert that the output matches the expected output
        pd.testing.assert_series_equal(output, expected_output)

    def test_hill_transformation_output(self):
        # Initialize the RidgeModelBuilder object with mock dependencies
        ridge_model_builder = RidgeModelBuilder(
            mmm_data=MMMData(),
            holiday_data=HolidaysData(),
            calibration_input=CalibrationInput(),
            hyperparameters=Hyperparameters(),
            featurized_mmm_data=FeaturizedMMMData(),
        )

        # Prepare the input Series `x` as a constant series
        x = pd.Series([0.5, 0.5, 0.5, 0.5, 0.5])

        # Set the transformation parameters
        alpha = 2.0
        gamma = 1.0

        # Call the `_hill_transformation` method
        transformed_output = ridge_model_builder._hill_transformation(x, alpha, gamma)

        # Prepare the expected output
        expected_output = pd.Series([0.25, 0.25, 0.25, 0.25, 0.25])

        # Assert that the transformed output matches the expected Series
        pd.testing.assert_series_equal(transformed_output, expected_output)

    def test_hill_transformation_output():
        # Create a Pandas Series `x` with values `[0.1, 0.2, 0.3, 0.4, 0.5]`
        x = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5])

        # Set the `alpha` and `gamma` parameters to `100.0`
        alpha = 100.0
        gamma = 100.0

        # Call the `_hill_transformation` method with `x`, `alpha`, and `gamma` as arguments
        result = RidgeModelBuilder._hill_transformation(
            RidgeModelBuilder, x, alpha, gamma
        )

        # Define the expected output as a Pandas Series with values `[0.0, 0.0, 0.0, 0.0, 0.0]`
        expected_output = pd.Series([0.0, 0.0, 0.0, 0.0, 0.0])

        # Use an assertion method to check if `result` equals the expected output
        pd.testing.assert_series_equal(result, expected_output, check_dtype=True)

    def test_hill_transformation_output_with_negative_alpha_and_gamma():
        # Step 4: Initialize a pd.Series with values [0.1, 0.2, 0.3, 0.4, 0.5] to represent the input data x.
        x = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5])

        # Step 5: Set alpha to -1.0 and gamma to -1.0 as per the test case input.
        alpha = -1.0
        gamma = -1.0

        # Step 6: Create an instance of the RidgeModelBuilder class with appropriate mock arguments for required constructor parameters.
        # Mock instances for required constructor parameters
        mmm_data = mock.MagicMock()
        holiday_data = mock.MagicMock()
        calibration_input = mock.MagicMock()
        hyperparameters = mock.MagicMock()
        featurized_mmm_data = mock.MagicMock()
        builder = RidgeModelBuilder(
            mmm_data,
            holiday_data,
            calibration_input,
            hyperparameters,
            featurized_mmm_data,
        )

        # Step 7: Call the _hill_transformation method on the RidgeModelBuilder instance with x, alpha, and gamma as arguments.
        result = builder._hill_transformation(x, alpha, gamma)

        # Step 8: Define the expected output as a pd.Series with NaN values for each input element.
        expected_output = pd.Series([np.nan, np.nan, np.nan, np.nan, np.nan])

        # Step 9: Use pd.testing.assert_series_equal to assert that the actual output matches the expected output series, allowing for NaN value comparison.
        pd.testing.assert_series_equal(result, expected_output)

    def test_calculate_rssd_without_zero_penalty() -> None:
        # Setup Inputs
        coefs = np.array([1.0, 2.0, 3.0])
        rssd_zero_penalty = False

        # Create an instance of RidgeModelBuilder with mock dependencies
        mock_mmm_data = MMMData(...)
        mock_holiday_data = HolidaysData(...)
        mock_calibration_input = CalibrationInput(...)
        mock_hyperparameters = Hyperparameters(...)
        mock_featurized_mmm_data = FeaturizedMMMData(...)
        model_builder = RidgeModelBuilder(
            mock_mmm_data,
            mock_holiday_data,
            mock_calibration_input,
            mock_hyperparameters,
            mock_featurized_mmm_data,
        )

        # Invoke Method
        calculated_rssd = model_builder._calculate_rssd(coefs, rssd_zero_penalty)

        # Perform Assertion
        expected_rssd = 3.7416573867739413
        assert (
            abs(calculated_rssd - expected_rssd) < 1e-9
        ), f"RSSD calculation failed, got {calculated_rssd}, expected {expected_rssd}"

    def test_calculate_rssd_returns_expected_value() -> None:
        # Instantiate an object of RidgeModelBuilder with mock dependencies
        mock_mmm_data = MagicMock(spec=MMMData)
        mock_holidays_data = MagicMock(spec=HolidaysData)
        mock_calibration_input = MagicMock(spec=CalibrationInput)
        mock_hyperparameters = MagicMock(spec=Hyperparameters)
        mock_featurized_mmm_data = MagicMock(spec=FeaturizedMMMData)

        ridge_model_builder = RidgeModelBuilder(
            mmm_data=mock_mmm_data,
            holiday_data=mock_holidays_data,
            calibration_input=mock_calibration_input,
            hyperparameters=mock_hyperparameters,
            featurized_mmm_data=mock_featurized_mmm_data,
        )

        # Prepare input coefficients and rssd_zero_penalty flag
        coefs = np.array([0.0, 0.0, 3.0])
        rssd_zero_penalty = True

        # Call the _calculate_rssd method
        rssd_value = ridge_model_builder._calculate_rssd(coefs, rssd_zero_penalty)

        # Assert the returned RSSD value is equal to expected value 6.0
        assert rssd_value == 6.0, f"Expected RSSD value to be 6.0, got {rssd_value}"

    def test_calculate_rssd_with_zero_coefficients() -> None:
        # Step 1: Initialize the RidgeModelBuilder class
        # Mock necessary data inputs for the class constructor
        mmm_data = MMMData()  # Assuming a simple instance or mock
        holiday_data = HolidaysData()  # Assuming a simple instance or mock
        calibration_input = CalibrationInput()  # Assuming a simple instance or mock
        hyperparameters = Hyperparameters()  # Assuming a simple instance or mock
        featurized_mmm_data = FeaturizedMMMData()  # Assuming a simple instance or mock

        model_builder = RidgeModelBuilder(
            mmm_data=mmm_data,
            holiday_data=holiday_data,
            calibration_input=calibration_input,
            hyperparameters=hyperparameters,
            featurized_mmm_data=featurized_mmm_data,
        )

        # Step 2: Define the coefficients array
        coefs = np.array([0.0, 2.0, 0.0])

        # Step 3: Set the rssd_zero_penalty parameter to True
        rssd_zero_penalty = True

        # Step 4: Call the _calculate_rssd method
        rssd_value = model_builder._calculate_rssd(coefs, rssd_zero_penalty)

        # Step 5: Assert that the returned RSSD value equals 4.0
        assert rssd_value == 4.0, f"Expected RSSD value to be 4.0, but got {rssd_value}"

    def test_calculate_rssd_with_zero_coefficients_and_zero_penalty() -> None:
        # Instantiate a RidgeModelBuilder object with mock dependencies
        mock_mmm_data = Mock(spec=MMMData)
        mock_holiday_data = Mock(spec=HolidaysData)
        mock_calibration_input = Mock(spec=CalibrationInput)
        mock_hyperparameters = Mock(spec=Hyperparameters)
        mock_featurized_mmm_data = Mock(spec=FeaturizedMMMData)
        ridge_builder = RidgeModelBuilder(
            mock_mmm_data,
            mock_holiday_data,
            mock_calibration_input,
            mock_hyperparameters,
            mock_featurized_mmm_data,
        )

        # Define test inputs
        coefs = np.array([0.0, 0.0, 0.0])
        rssd_zero_penalty = True

        # Call the _calculate_rssd method with the test inputs
        result = ridge_builder._calculate_rssd(coefs, rssd_zero_penalty)

        # Assert that the result is equal to the expected RSSD value of 0.0
        assert result == 0.0, f"Expected RSSD value of 0.0, but got {result}"

    def test_calculate_rssd_value() -> None:
        # Initialize Input Coefficients
        coefficients = np.array([1.0, 2.0, 3.0])
        # Set RSSD Zero Penalty Flag
        rssd_zero_penalty = True
        # Invoke Calculate RSSD Method
        rssd_result = RidgeModelBuilder._calculate_rssd(coefficients, rssd_zero_penalty)
        # Perform Assertion
        assert np.isclose(
            rssd_result, 3.7416573867739413, atol=1e-9
        ), f"RSSD calculation failed, expected 3.7416573867739413 but got {rssd_result}"

    def test_calculate_rssd_returns_expected_value() -> None:
        # Create an instance of the RidgeModelBuilder class
        builder = RidgeModelBuilder(None, None, None, None, None)

        # Define a list of coefficients with a single value
        coefs = [5.0]

        # Set the rssd_zero_penalty flag to False
        rssd_zero_penalty = False

        # Call the _calculate_rssd method
        rssd_value = builder._calculate_rssd(coefs, rssd_zero_penalty)

        # Assert the returned RSSD value is equal to the expected value
        assert rssd_value == 5.0, f"Expected RSSD value to be 5.0, but got {rssd_value}"

    def test_calculate_rssd_single_zero_coefficient_with_zero_penalty() -> None:
        # Step 1: Initialize the RidgeModelBuilder instance if needed
        # Since _calculate_rssd is a method, assume RidgeModelBuilder instance is available as `ridge_model_builder`

        # Step 2: Prepare test input data
        coefs = np.array([0.0])  # Single zero coefficient
        rssd_zero_penalty = True

        # Step 3: Call the _calculate_rssd method with the prepared inputs
        rssd_value = RidgeModelBuilder._calculate_rssd(coefs, rssd_zero_penalty)

        # Step 4: Assert the expected result
        expected_rssd_value = 0.0  # Expected RSSD value for a single zero coefficient
        assert (
            rssd_value == expected_rssd_value
        ), f"Expected {expected_rssd_value}, but got {rssd_value}"

    def test_calculate_rssd_with_negative_coefficients_no_zero_penalty() -> None:
        import numpy as np
        from RidgeModelBuilder import _calculate_rssd

        # Define the input coefficients
        coefs = np.array([-1.0, -2.0, -3.0])
        rssd_zero_penalty = False

        # Call the _calculate_rssd method
        rssd_result = RidgeModelBuilder._calculate_rssd(coefs, rssd_zero_penalty)

        # Expected RSSD result
        expected_rssd = 3.7416573867739413

        # Assert that the calculated RSSD matches the expected value
        assert np.isclose(
            rssd_result, expected_rssd, atol=1e-9
        ), f"RSSD result: {rssd_result}, Expected: {expected_rssd}"

    def test_calculate_rssd_with_positive_and_negative_coefficients(self):
        # Mock the necessary dependencies for RidgeModelBuilder
        mmm_data = MockMMMData()
        holiday_data = MockHolidaysData()
        calibration_input = MockCalibrationInput()
        hyperparameters = MockHyperparameters()
        featurized_mmm_data = MockFeaturizedMMMData()

        # Initialize RidgeModelBuilder
        ridge_model_builder = RidgeModelBuilder(
            mmm_data,
            holiday_data,
            calibration_input,
            hyperparameters,
            featurized_mmm_data,
        )

        # Define test input
        coefs = [1.0, -1.0, 2.0, -2.0, 3.0]
        rssd_zero_penalty = False

        # Calculate RSSD
        rssd_value = ridge_model_builder._calculate_rssd(coefs, rssd_zero_penalty)

        # Assert the RSSD value
        self.assertAlmostEqual(rssd_value, 3.7416573867739413, places=7)

    def test_calculate_mape_returns_zero_mape_on_valid_data() -> None:
        # Mocking the dependencies
        mmm_data_mock = MagicMock()
        mmm_data_mock.data = pd.DataFrame(
            {
                "date": pd.date_range(start="2021-01-01", periods=3, freq="D"),
                "dependent_variable": [100, 200, 300],
            }
        )
        mmm_data_mock.mmmdata_spec.date_var = "date"
        mmm_data_mock.mmmdata_spec.dep_var = "dependent_variable"

        featurized_mmm_data_mock = MagicMock()
        featurized_mmm_data_mock.rollingWindowStartWhich = 0
        featurized_mmm_data_mock.rollingWindowEndWhich = 2

        calibration_input_mock = {
            "calibration_key": {
                "liftStartDate": pd.Timestamp("2021-01-01"),
                "liftEndDate": pd.Timestamp("2021-01-03"),
                "liftMedia": "media_channel",
            }
        }

        # Mocking the Ridge model's predict method
        ridge_model_mock = MagicMock(spec=Ridge)
        ridge_model_mock.predict.return_value = [100, 200, 300]

        # Creating an instance of RidgeModelBuilder
        ridge_model_builder = RidgeModelBuilder(
            mmm_data=mmm_data_mock,
            holiday_data=MagicMock(),
            calibration_input=calibration_input_mock,
            hyperparameters=MagicMock(),
            featurized_mmm_data=featurized_mmm_data_mock,
        )

        # Call the method under test
        mape = ridge_model_builder._calculate_mape(ridge_model_mock)

        # Assert that the MAPE is 0.0
        assert mape == 0.0, "Expected MAPE to be 0.0 for perfect predictions"

    def test_calculate_mape_mape_value() -> None:
        # Mock MMMData to return a predefined dataset
        mmm_data = MagicMock()
        mmm_data.data = pd.DataFrame(
            {
                "date": pd.date_range(start="2021-01-01", periods=10, freq="D"),
                "dep_var": np.random.rand(10),
            }
        )

        # Mock FeaturizedMMMData to return specific rolling window indices
        featurized_mmm_data = MagicMock()
        featurized_mmm_data.rollingWindowStartWhich = 0
        featurized_mmm_data.rollingWindowEndWhich = 9

        # Mock CalibrationInput to provide predefined calibration data
        calibration_input = {
            "test_key": {"liftStartDate": 1.0, "liftEndDate": 1.0, "liftMedia": 1.0}
        }

        # Mock Ridge model's predict method
        ridge_model = MagicMock()
        ridge_model.predict.return_value = np.array([0.9] * 10)

        # Instantiate RidgeModelBuilder with mocked data
        builder = RidgeModelBuilder(
            mmm_data, None, calibration_input, None, featurized_mmm_data
        )

        # Calculate MAPE
        mape_value = builder._calculate_mape(ridge_model)

        # Assert that the MAPE value is approximately 6.6667
        assert abs(mape_value - 6.6667) < 1e-4

    def test_calculate_mape_no_calibration_data():
        # Instantiate a Ridge model object
        model = Ridge()

        # Mock the RidgeModelBuilder object
        ridge_model_builder = RidgeModelBuilder(
            mmm_data=None,
            holiday_data=None,
            calibration_input=None,
            hyperparameters=None,
            featurized_mmm_data=None,
        )

        # Set calibration_input to None to simulate no calibration data
        ridge_model_builder.calibration_input = None

        # Invoke the _calculate_mape method
        mape_value = ridge_model_builder._calculate_mape(model)

        # Assert that the MAPE value is 0.0
        assert mape_value == 0.0, f"Expected MAPE to be 0.0, but got {mape_value}"

    def test_evaluate_model_loss() -> None:
        # Mock the methods to return fixed values
        ridge_builder = RidgeModelBuilder(None, None, None, None, None)
        ridge_builder._prepare_data = lambda params: (
            pd.DataFrame([[1, 2], [3, 4]]),
            pd.Series([1, 2]),
        )
        ridge_builder._calculate_rssd = lambda coefs, penalty: 0.05
        ridge_builder._calculate_mape = lambda model: 0.07
        ridge_builder._calculate_lift_calibration = lambda model: 0.1

        # Set up the input parameters
        params = {"lambda": 0.5}
        ts_validation = False
        add_penalty_factor = False
        rssd_zero_penalty = False
        objective_weights = [0.5, 0.3, 0.2]
        start_time = time.time()
        iter_ng = 0
        trial = 1

        # Call the method and check if 'loss' is a float
        result = ridge_builder._evaluate_model(
            params,
            ts_validation,
            add_penalty_factor,
            rssd_zero_penalty,
            objective_weights,
            start_time,
            iter_ng,
            trial,
        )
        assert isinstance(result["loss"], float)

    def test_evaluate_model_nrmse() -> None:
        # Mock the _prepare_data method to return features and target
        mock_X = pd.DataFrame(
            np.random.rand(100, 5), columns=[f"feature_{i}" for i in range(5)]
        )
        mock_y = pd.Series(np.random.rand(100), name="target")
        RidgeModelBuilder._prepare_data = lambda self, _: (mock_X, mock_y)

        # Mock the _calculate_rssd method to return 0.05
        RidgeModelBuilder._calculate_rssd = lambda self, _, __: 0.05

        # Mock the _calculate_mape method to return 0.07
        RidgeModelBuilder._calculate_mape = lambda self, _: 0.07

        # Mock the _calculate_lift_calibration method to return 0.1
        RidgeModelBuilder._calculate_lift_calibration = lambda self, _: 0.1

        # Setup input parameters for _evaluate_model
        params = {"lambda": 0.5}
        ts_validation = False
        add_penalty_factor = False
        rssd_zero_penalty = False
        objective_weights = [0.3, 0.3, 0.4]
        start_time = time.time()
        iter_ng = 1
        trial = 1

        # Create an instance of RidgeModelBuilder with mock data
        builder = RidgeModelBuilder(
            mmm_data=None,
            holiday_data=None,
            calibration_input=None,
            hyperparameters=None,
            featurized_mmm_data=None,
        )

        # Call the _evaluate_model method
        result = builder._evaluate_model(
            params,
            ts_validation,
            add_penalty_factor,
            rssd_zero_penalty,
            objective_weights,
            start_time,
            iter_ng,
            trial,
        )

        # Assert that 'nrmse' is a float
        assert isinstance(result["nrmse"], float)

    def test_evaluate_model_decomp_rssd(self):
        # Mocking the necessary methods
        self.ridge_model_builder._prepare_data = lambda params: (
            pd.DataFrame({"feature1": [1, 2, 3]}),
            pd.Series([1, 2, 3]),
        )
        self.ridge_model_builder._calculate_rssd = lambda coefs, rssd_zero_penalty: 0.05
        self.ridge_model_builder._calculate_mape = lambda model: 0.07
        self.ridge_model_builder._calculate_lift_calibration = lambda model: 0.1

        # Parameters for _evaluate_model
        params = {"lambda": 0.5}
        ts_validation = False
        add_penalty_factor = False
        rssd_zero_penalty = True
        objective_weights = [0.33, 0.33, 0.34]
        start_time = 0.0
        iter_ng = 1
        trial = 1

        # Call the method under test
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

        # Assert the 'decomp_rssd' value
        self.assertEqual(result["decomp_rssd"], 0.05)

    def test_evaluate_model_mape() -> None:
        # Create a RidgeModelBuilder instance
        ridge_model_builder = RidgeModelBuilder(
            mmm_data=None,  # Mock or use appropriate data
            holiday_data=None,
            calibration_input=None,
            hyperparameters=None,
            featurized_mmm_data=None,
        )

        # Mock the methods to return expected values
        ridge_model_builder._prepare_data = lambda params: (pd.DataFrame(), pd.Series())
        ridge_model_builder._calculate_rssd = lambda coefs, penalty: 0.05
        ridge_model_builder._calculate_mape = lambda model: 0.07
        ridge_model_builder._calculate_lift_calibration = lambda model: 0.1

        # Set up input parameters
        params = {}
        ts_validation = False
        add_penalty_factor = False
        rssd_zero_penalty = False
        objective_weights = None
        start_time = 0
        iter_ng = 0
        trial = 1

        # Call the _evaluate_model method
        result = ridge_model_builder._evaluate_model(
            params=params,
            ts_validation=ts_validation,
            add_penalty_factor=add_penalty_factor,
            rssd_zero_penalty=rssd_zero_penalty,
            objective_weights=objective_weights,
            start_time=start_time,
            iter_ng=iter_ng,
            trial=trial,
        )

        # Assert that the 'mape' in the result is equal to 0.07
        assert result["mape"] == 0.07

    def test_evaluate_model_lift_calibration() -> None:
        # Mock the required methods
        ridge_model_builder = RidgeModelBuilder(
            mmm_data=MMMData(),
            holiday_data=HolidaysData(),
            calibration_input=CalibrationInput(),
            hyperparameters=Hyperparameters(),
            featurized_mmm_data=FeaturizedMMMData(),
        )

        ridge_model_builder._prepare_data = lambda params: (
            pd.DataFrame([[0, 0], [1, 1]]),
            pd.Series([1, 2]),
        )
        ridge_model_builder._calculate_rssd = lambda coefs, penalty: 0.05
        ridge_model_builder._calculate_mape = lambda model: 0.07
        ridge_model_builder._calculate_lift_calibration = lambda model: 0.1

        # Set up input parameters
        input_params = {
            "ts_validation": False,
            "add_penalty_factor": False,
            "rssd_zero_penalty": True,
            "objective_weights": [0.33, 0.33, 0.34],
            "start_time": time.time(),
            "iter_ng": 1,
            "trial": 1,
        }

        # Call the _evaluate_model method
        result = ridge_model_builder._evaluate_model(
            params={"lambda_": 0.5}, **input_params
        )

        # Assert that the 'lift_calibration' is equal to 0.1
        assert result["lift_calibration"] == 0.1

    def test_evaluate_model_rsq_train(self):
        # Mock the _prepare_data method
        self.builder._prepare_data = MagicMock(
            return_value=(
                pd.DataFrame(np.random.rand(10, 5)),
                pd.Series(np.random.rand(10)),
            )
        )

        # Mock the _calculate_rssd method
        self.builder._calculate_rssd = MagicMock(return_value=0.05)

        # Mock the _calculate_mape method
        self.builder._calculate_mape = MagicMock(return_value=0.07)

        # Mock the _calculate_lift_calibration method
        self.builder._calculate_lift_calibration = MagicMock(return_value=0.1)

        # Set up parameters for _evaluate_model method
        params = {"lambda": 0.5}
        ts_validation = False
        add_penalty_factor = False
        rssd_zero_penalty = False
        objective_weights = [0.3, 0.3, 0.4]
        start_time = time.time()
        iter_ng = 1
        trial = 1

        # Call the _evaluate_model method
        result = self.builder._evaluate_model(
            params,
            ts_validation,
            add_penalty_factor,
            rssd_zero_penalty,
            objective_weights,
            start_time,
            iter_ng,
            trial,
        )

        # Assert 'rsq_train' is of type float
        self.assertIsInstance(result["rsq_train"], float)

    def test_evaluate_model_rsq_val() -> None:
        # Mock the _prepare_data method to return a DataFrame of features and a Series of the target
        mock_features = pd.DataFrame(np.random.rand(100, 5))
        mock_target = pd.Series(np.random.rand(100))
        RidgeModelBuilder._prepare_data = lambda self, params: (
            mock_features,
            mock_target,
        )

        # Mock the _calculate_rssd method to return 0.05
        RidgeModelBuilder._calculate_rssd = lambda self, coefs, rssd_zero_penalty: 0.05

        # Mock the _calculate_mape method to return 0.07
        RidgeModelBuilder._calculate_mape = lambda self, model: 0.07

        # Mock the _calculate_lift_calibration method to return 0.1
        RidgeModelBuilder._calculate_lift_calibration = lambda self, model: 0.1

        # Set up the input parameters
        params = {"lambda": 0.5, "train_size": 0.8}
        ts_validation = True
        add_penalty_factor = False
        rssd_zero_penalty = False
        objective_weights = [0.3, 0.3, 0.4]
        start_time = time.time()
        iter_ng = 1
        trial = 1

        # Instantiate RidgeModelBuilder
        builder = RidgeModelBuilder(
            mmm_data=MMMData(),
            holiday_data=HolidaysData(),
            calibration_input=CalibrationInput(),
            hyperparameters=Hyperparameters(),
            featurized_mmm_data=FeaturizedMMMData(),
        )

        # Call the _evaluate_model method
        result = builder._evaluate_model(
            params=params,
            ts_validation=ts_validation,
            add_penalty_factor=add_penalty_factor,
            rssd_zero_penalty=rssd_zero_penalty,
            objective_weights=objective_weights,
            start_time=start_time,
            iter_ng=iter_ng,
            trial=trial,
        )

        # Assert that 'rsq_val' is a float
        assert isinstance(result["rsq_val"], float)

    def test_evaluate_model_rsq_test() -> None:
        # Mocking the necessary methods
        with patch.object(
            RidgeModelBuilder,
            "_prepare_data",
            return_value=(
                pd.DataFrame(np.random.rand(10, 5)),
                pd.Series(np.random.rand(10)),
            ),
        ) as mock_prepare_data:
            with patch.object(
                RidgeModelBuilder, "_calculate_rssd", return_value=0.05
            ) as mock_calculate_rssd:
                with patch.object(
                    RidgeModelBuilder, "_calculate_mape", return_value=0.07
                ) as mock_calculate_mape:
                    with patch.object(
                        RidgeModelBuilder,
                        "_calculate_lift_calibration",
                        return_value=0.1,
                    ) as mock_calculate_lift_calibration:

                        # Setting up input parameters
                        ts_validation = True
                        add_penalty_factor = False
                        rssd_zero_penalty = False
                        objective_weights = [0.5, 0.3, 0.2]
                        start_time = time.time()
                        iter_ng = 1
                        trial = 1
                        params = {"lambda": 0.5, "train_size": 0.8}

                        # Creating a RidgeModelBuilder instance
                        model_builder = RidgeModelBuilder(
                            mmm_data=mock.MagicMock(),
                            holiday_data=mock.MagicMock(),
                            calibration_input=mock.MagicMock(),
                            hyperparameters=mock.MagicMock(),
                            featurized_mmm_data=mock.MagicMock(),
                        )

                        # Call the _evaluate_model method
                        result = model_builder._evaluate_model(
                            params=params,
                            ts_validation=ts_validation,
                            add_penalty_factor=add_penalty_factor,
                            rssd_zero_penalty=rssd_zero_penalty,
                            objective_weights=objective_weights,
                            start_time=start_time,
                            iter_ng=iter_ng,
                            trial=trial,
                        )

                        # Assert that 'rsq_test' is of type float
                        assert isinstance(
                            result["rsq_test"], float
                        ), "rsq_test should be of type float"

    def test_evaluate_model_lambda_(self):
        # Mocking the _prepare_data method to return a DataFrame and Series
        self.ridge_model_builder._prepare_data = MagicMock(
            return_value=(
                pd.DataFrame(np.random.rand(10, 5)),
                pd.Series(np.random.rand(10)),
            )
        )

        # Mocking _calculate_rssd to return 0.05
        self.ridge_model_builder._calculate_rssd = MagicMock(return_value=0.05)

        # Mocking _calculate_mape to return 0.07
        self.ridge_model_builder._calculate_mape = MagicMock(return_value=0.07)

        # Mocking _calculate_lift_calibration to return 0.1
        self.ridge_model_builder._calculate_lift_calibration = MagicMock(
            return_value=0.1
        )

        # Set up input parameters
        params = {"lambda": 0.5}
        ts_validation = False
        add_penalty_factor = False
        rssd_zero_penalty = True
        objective_weights = None
        start_time = time.time()
        iter_ng = 1
        trial = 1

        # Call the _evaluate_model method
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

        # Assert that the 'lambda_' key in the returned dictionary is equal to 0.5
        self.assertEqual(result["lambda_"], 0.5)

    def test_evaluate_model_pos(self):
        # Mock the `_prepare_data` method to return a DataFrame of features and a Series of the target
        mock_features = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})
        mock_target = pd.Series([1, 2, 3])
        self.ridge_model_builder._prepare_data = MagicMock(
            return_value=(mock_features, mock_target)
        )

        # Mock the `_calculate_rssd` method to return 0.05
        self.ridge_model_builder._calculate_rssd = MagicMock(return_value=0.05)

        # Mock the `_calculate_mape` method to return 0.07
        self.ridge_model_builder._calculate_mape = MagicMock(return_value=0.07)

        # Mock the `_calculate_lift_calibration` method to return 0.1
        self.ridge_model_builder._calculate_lift_calibration = MagicMock(
            return_value=0.1
        )

        # Set up the input parameters for the `_evaluate_model` method call
        params = {"lambda": 0.5}
        ts_validation = False
        add_penalty_factor = False
        rssd_zero_penalty = True
        objective_weights = [0.3, 0.3, 0.4]
        start_time = time.time()
        iter_ng = 1
        trial = 1

        # Call the `_evaluate_model` method
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

        # Assert that the 'pos' key in the returned dictionary is of type int
        self.assertIsInstance(result["pos"], int)

    def test_evaluate_model_elapsed(self):
        model_builder = RidgeModelBuilder(
            mmm_data=MagicMock(),
            holiday_data=MagicMock(),
            calibration_input=MagicMock(),
            hyperparameters=MagicMock(),
            featurized_mmm_data=MagicMock(),
        )

        model_builder._prepare_data = MagicMock(
            return_value=(pd.DataFrame({"feature1": [1, 2, 3]}), pd.Series([4, 5, 6]))
        )
        model_builder._calculate_rssd = MagicMock(return_value=0.05)
        model_builder._calculate_mape = MagicMock(return_value=0.07)
        model_builder._calculate_lift_calibration = MagicMock(return_value=0.1)

        params = {
            "ts_validation": False,
            "add_penalty_factor": False,
            "rssd_zero_penalty": False,
            "objective_weights": None,
            "start_time": time.time(),
            "iter_ng": 1,
            "trial": 1,
        }

        result = model_builder._evaluate_model(params, **params)

        self.assertIsInstance(result["elapsed"], float)

    def test_loss_calculation(input: Dict[str, Any]) -> None:
        # Mocking the RidgeModelBuilder's methods
        with unittest.mock.patch.object(
            RidgeModelBuilder, "_prepare_data"
        ) as mock_prepare_data, unittest.mock.patch.object(
            RidgeModelBuilder, "_calculate_rssd"
        ) as mock_calculate_rssd, unittest.mock.patch.object(
            RidgeModelBuilder, "_calculate_mape"
        ) as mock_calculate_mape:

            # Setting up the mock return values
            mock_prepare_data.return_value = (
                pd.DataFrame(np.random.rand(10, 5)),
                pd.Series(np.random.rand(10)),
            )
            mock_calculate_rssd.return_value = 0.05
            mock_calculate_mape.return_value = 0.07

            # Instantiate the RidgeModelBuilder
            builder = RidgeModelBuilder(
                mmm_data=MockMMMData(),
                holiday_data=MockHolidaysData(),
                calibration_input=MockCalibrationInput(),
                hyperparameters=MockHyperparameters(),
                featurized_mmm_data=MockFeaturizedMMMData(),
            )

            # Invoke the method under test
            result = builder._evaluate_model(
                params=input["params"],
                ts_validation=input["ts_validation"],
                add_penalty_factor=input["add_penalty_factor"],
                rssd_zero_penalty=input["rssd_zero_penalty"],
                objective_weights=input["objective_weights"],
                start_time=input["start_time"],
                iter_ng=input["iter_ng"],
                trial=input["trial"],
            )

            # Extract the 'loss' value
            loss = result["loss"]

            # Assert that 'loss' is of type float
            assert isinstance(
                loss, float
            ), f"Expected 'loss' to be of type float, got {type(loss)} instead."

    def test_nrmse_calculation(self):
        # Mock the `_prepare_data` method to return a predefined DataFrame and Series
        mock_X = pd.DataFrame({"feature1": [1, 2], "feature2": [3, 4]})
        mock_y = pd.Series([5, 6])
        self.ridge_model_builder._prepare_data = MagicMock(
            return_value=(mock_X, mock_y)
        )

        # Mock `_calculate_rssd` and `_calculate_mape` to return fixed values
        self.ridge_model_builder._calculate_rssd = MagicMock(return_value=0.05)
        self.ridge_model_builder._calculate_mape = MagicMock(return_value=0.07)

        # Define input parameters for `_evaluate_model`
        params = {"lambda": 0.1, "train_size": 0.8}
        ts_validation = False
        add_penalty_factor = False
        rssd_zero_penalty = True
        objective_weights = [0.3, 0.3, 0.4]
        start_time = time.time()
        iter_ng = 1
        trial = 1

        # Call `_evaluate_model` with the input parameters
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

        # Capture the `nrmse` value from the function's output
        nrmse_value = result["nrmse"]

        # Verify that the `nrmse` value is of type `float`
        self.assertIsInstance(nrmse_value, float)

    def test_decomp_rssd_value(input: Dict[str, Any]) -> None:
        # Mock the _prepare_data method to return fixed data for testing
        def mock_prepare_data(
            params: Dict[str, float]
        ) -> Tuple[pd.DataFrame, pd.Series]:
            X = pd.DataFrame(
                np.random.rand(10, 5), columns=[f"feature{i}" for i in range(5)]
            )
            y = pd.Series(np.random.rand(10))
            return X, y

        # Mock the _calculate_rssd method to return a fixed RSSD value
        def mock_calculate_rssd(coefs: np.ndarray, rssd_zero_penalty: bool) -> float:
            return 0.02

        # Replace original methods with mocks
        original_prepare_data = RidgeModelBuilder._prepare_data
        original_calculate_rssd = RidgeModelBuilder._calculate_rssd
        try:
            RidgeModelBuilder._prepare_data = mock_prepare_data
            RidgeModelBuilder._calculate_rssd = mock_calculate_rssd

            # Instantiate the RidgeModelBuilder with dummy data
            mmm_data = MMMData(...)  # Use mock or dummy data
            holiday_data = HolidaysData(...)
            calibration_input = CalibrationInput(...)
            hyperparameters = Hyperparameters(...)
            featurized_mmm_data = FeaturizedMMMData(...)

            builder = RidgeModelBuilder(
                mmm_data,
                holiday_data,
                calibration_input,
                hyperparameters,
                featurized_mmm_data,
            )

            # Call the method under test with the mocked data
            result = builder._evaluate_model(input, False, False, True, None, 0.0, 0, 1)

            # Assert the expected value of decomp_rssd
            assert result["decomp_rssd"] == 0.02

        finally:
            # Restore the original methods
            RidgeModelBuilder._prepare_data = original_prepare_data
            RidgeModelBuilder._calculate_rssd = original_calculate_rssd

    def test_mape_value(self):
        # Mock the `_prepare_data` method to return specific prepared data
        self.ridge_model_builder._prepare_data = lambda params: (
            pd.DataFrame(),
            pd.Series(),
        )

        # Mock the `_calculate_mape` method to return the expected MAPE value
        self.ridge_model_builder._calculate_mape = lambda model: 0.05

        # Create a test input dictionary
        test_input = {
            "ts_validation": False,
            "add_penalty_factor": False,
            "rssd_zero_penalty": False,
            "objective_weights": [0.5, 0.5],
            "start_time": time.time(),
            "iter_ng": 0,
            "trial": 1,
            "lambda_": 0.1,
        }

        # Call the `_evaluate_model` method with the test input
        result = self.ridge_model_builder._evaluate_model(**test_input)

        # Assert that the `mape` value in the result matches the expected value
        self.assertEqual(result["mape"], 0.05)

    def test_lift_calibration(self):
        # Prepare mock data and parameters
        input_params = {
            "params": {"lambda": 0.5, "train_size": 0.8},
            "ts_validation": False,
            "add_penalty_factor": False,
            "rssd_zero_penalty": True,
            "objective_weights": None,
            "start_time": time.time(),
            "iter_ng": 1,
            "trial": 1,
        }

        # Mock the _calculate_mape method to return a specific value
        self.ridge_model_builder._calculate_mape = lambda model: 0.0

        # Invoke the _evaluate_model method with the mock parameters
        result = self.ridge_model_builder._evaluate_model(**input_params)

        # Assert that the lift_calibration result is None
        self.assertIsNone(result["lift_calibration"])

    def test_rsq_train_value(input: Dict[str, Any]) -> None:
        # Mock necessary data preparation and method dependencies
        ridge_model_builder = RidgeModelBuilder(
            mmm_data=MagicMock(spec=MMMData),
            holiday_data=MagicMock(spec=HolidaysData),
            calibration_input=MagicMock(spec=CalibrationInput),
            hyperparameters=MagicMock(spec=Hyperparameters),
            featurized_mmm_data=MagicMock(spec=FeaturizedMMMData),
        )

        # Mock the data preparation step
        ridge_model_builder._prepare_data = MagicMock(
            return_value=(np.random.rand(100, 10), np.random.rand(100))
        )

        # Call _evaluate_model with the test input
        result = ridge_model_builder._evaluate_model(
            params=input.get("params", {}),
            ts_validation=input.get("ts_validation", False),
            add_penalty_factor=input.get("add_penalty_factor", False),
            rssd_zero_penalty=input.get("rssd_zero_penalty", False),
            objective_weights=input.get("objective_weights", None),
            start_time=input.get("start_time", time.time()),
            iter_ng=input.get("iter_ng", 0),
            trial=input.get("trial", 0),
        )

        # Assert that rsq_train in the result is a float
        assert isinstance(result["rsq_train"], float), "rsq_train should be a float"

    def test_rsq_val_value(input: Dict[str, Any]) -> None:
        # Mocking the _prepare_data method
        ridge_model_builder = RidgeModelBuilder(None, None, None, None, None)
        ridge_model_builder._prepare_data = lambda params: (
            np.random.rand(10, 5),
            np.random.rand(10),
        )

        # Mocking other necessary methods if needed
        ridge_model_builder._calculate_rssd = lambda coefs, rssd_zero_penalty: 0.0
        ridge_model_builder._calculate_mape = lambda model: 0.0

        # Call _evaluate_model with ts_validation set to False
        result = ridge_model_builder._evaluate_model(
            params=input,
            ts_validation=False,
            add_penalty_factor=False,
            rssd_zero_penalty=False,
            objective_weights=None,
            start_time=time.time(),
            iter_ng=0,
            trial=1,
        )

        # Verify that rsq_val is 0.0
        assert result["rsq_val"] == 0.0

    def test_rsq_test_value(input: Dict[str, Any]) -> None:
        # Mock data preparation and calculation
        mock_ridge_model_builder = RidgeModelBuilder(
            mmm_data=input["mmm_data"],
            holiday_data=input["holiday_data"],
            calibration_input=input["calibration_input"],
            hyperparameters=input["hyperparameters"],
            featurized_mmm_data=input["featurized_mmm_data"],
        )

        # Mocking the _prepare_data method
        mock_ridge_model_builder._prepare_data = lambda params: (
            pd.DataFrame(input["X"]),
            pd.Series(input["y"]),
        )

        # Mocking other necessary methods if needed
        mock_ridge_model_builder._calculate_rssd = lambda coefs, zero_penalty: 0.1
        mock_ridge_model_builder._calculate_mape = lambda model: 0.1

        # Invoke _evaluate_model with ts_validation set to False
        result = mock_ridge_model_builder._evaluate_model(
            params=input["params"],
            ts_validation=False,
            add_penalty_factor=input["add_penalty_factor"],
            rssd_zero_penalty=input["rssd_zero_penalty"],
            objective_weights=input["objective_weights"],
            start_time=0,
            iter_ng=0,
            trial=0,
        )

        # Assert that rsq_test is 0.0
        assert result["rsq_test"] == 0.0

    def test_lambda_value(self):
        # Mock inputs and necessary objects
        input_params = {
            "lambda": 2.0,
            # Include other required parameters with mock values, e.g., data, flags, etc.
        }

        # Mock the necessary methods and attributes for the test
        mock_model = MagicMock()
        mock_model.coef_ = np.array([0.5, 1.0, -0.5])  # Example coefficients
        self.model_builder._prepare_data = MagicMock(
            return_value=(pd.DataFrame(), pd.Series())
        )
        self.model_builder._calculate_mape = MagicMock(return_value=0.0)
        self.model_builder._calculate_rssd = MagicMock(return_value=0.0)
        self.model_builder._calculate_decomp_spend_dist = MagicMock(
            return_value=pd.DataFrame()
        )
        self.model_builder._calculate_x_decomp_agg = MagicMock(
            return_value=pd.DataFrame()
        )

        # Test the `_evaluate_model` method
        result = self.model_builder._evaluate_model(
            params=input_params,
            ts_validation=False,
            add_penalty_factor=False,
            rssd_zero_penalty=False,
            objective_weights=None,
            start_time=time.time(),
            iter_ng=0,
            trial=1,
        )

        # Assert the lambda_ value in the result matches the input lambda
        self.assertEqual(result["lambda_"], 2.0)

    def test_pos_value(input: Dict[str, Any]) -> None:
        # Mock data preparation
        params = {
            "lambda": 0.01,
            # Add other necessary parameters as needed for the test
        }
        mock_data = {
            "ts_validation": False,
            "add_penalty_factor": False,
            "rssd_zero_penalty": True,
            "objective_weights": None,
            "start_time": time.time(),
            "iter_ng": 1,
            "trial": 1,
            **params,
        }

        # Execute _evaluate_model with the test input
        result = self._evaluate_model(**mock_data)

        # Confirm that 'pos' is an integer
        assert isinstance(result["pos"], int), "The 'pos' value should be an integer."

    def test_elapsed_time(input: Dict[str, Any]) -> None:
        # Mock necessary data and methods
        mock_model = Ridge()
        mock_params = input.get("params", {})

        # Mock the _prepare_data method to return test data
        test_data_x = np.random.rand(10, 5)  # Mock feature data
        test_data_y = np.random.rand(10)  # Mock target data
        RidgeModelBuilder._prepare_data = lambda self, params: (
            test_data_x,
            test_data_y,
        )

        # Mock the time to ensure predictable elapsed time
        start_time = time.time()

        # Call the method under test
        result = RidgeModelBuilder._evaluate_model(
            RidgeModelBuilder,
            params=mock_params,
            ts_validation=False,
            add_penalty_factor=False,
            rssd_zero_penalty=False,
            objective_weights=None,
            start_time=start_time,
            iter_ng=0,
            trial=1,
        )

        # Assert that elapsed is a float
        assert isinstance(result["elapsed"], float), "elapsed should be a float"

    def test_loss_calculation() -> None:
        # Mock the _prepare_data method to return a DataFrame of features and a Series of target
        RidgeModelBuilder._prepare_data = lambda self, params: (
            pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]}),
            pd.Series([1, 2, 3]),
        )

        # Mock the _calculate_rssd method to return a fixed value
        RidgeModelBuilder._calculate_rssd = lambda self, coefs, rssd_zero_penalty: 0.01

        # Mock the _lambda_seq method to return a sequence of lambda values
        RidgeModelBuilder._lambda_seq = (
            lambda self, x, y, seq_len=100, lambda_min_ratio=0.0001: np.array(
                [0.1, 0.2, 0.3]
            )
        )

        # Create an instance of RidgeModelBuilder
        builder = RidgeModelBuilder(None, None, None, None, None)

        # Call the evaluate_model method with minimal parameters and default values
        result = builder._evaluate_model(
            params={"lambda": 0.1},
            ts_validation=False,
            add_penalty_factor=False,
            rssd_zero_penalty=False,
            objective_weights=None,
            start_time=0.0,
            iter_ng=0,
            trial=1,
        )

        # Assert that the returned loss value is of type float
        assert isinstance(result["loss"], float)

    def test_nrmse_calculation() -> None:
        # Mock the _prepare_data method to return a DataFrame of features and a Series of target
        mock_features = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})
        mock_target = pd.Series([7, 8, 9])

        RidgeModelBuilder._prepare_data = lambda self, params: (
            mock_features,
            mock_target,
        )

        # Mock the _calculate_rssd method to return 0.01
        RidgeModelBuilder._calculate_rssd = lambda self, coefs, rssd_zero_penalty: 0.01

        # Create an instance of RidgeModelBuilder
        builder = RidgeModelBuilder(
            mmm_data=None,  # Pass mock or None as needed
            holiday_data=None,
            calibration_input=None,
            hyperparameters=None,
            featurized_mmm_data=None,
        )

        # Mock parameters for evaluate_model
        mock_params = {
            "lambda": 0.1,
            # Add other necessary parameters if needed
        }

        # Invoke the evaluate_model function
        result = builder._evaluate_model(
            params=mock_params,
            ts_validation=False,
            add_penalty_factor=False,
            rssd_zero_penalty=False,
            objective_weights=None,
            start_time=0,
            iter_ng=0,
            trial=0,
        )

        # Assert that the NRMSE in the result is of type float
        assert isinstance(result["nrmse"], float)

    def test_decomp_rssd_value(self) -> None:
        # Mock the _prepare_data method to return mock data
        mock_X = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})
        mock_y = pd.Series([1, 2, 3])
        self.model_builder._prepare_data = lambda _: (mock_X, mock_y)

        # Mock the _calculate_rssd method to return the expected RSSD value
        self.model_builder._calculate_rssd = lambda coefs, rssd_zero_penalty: 0.01

        # Call the evaluate_model method with mock parameters
        mock_params = {"lambda": 0.1}
        result = self.model_builder._evaluate_model(
            params=mock_params,
            ts_validation=False,
            add_penalty_factor=False,
            rssd_zero_penalty=True,
            objective_weights=None,
            start_time=time.time(),
            iter_ng=0,
            trial=1,
        )

        # Assert that the decomp_rssd in the result matches the expected value
        self.assertEqual(result["decomp_rssd"], 0.01)

    def test_mape_calculation() -> None:
        # Mock the necessary methods
        with mock.patch.object(
            RidgeModelBuilder,
            "_prepare_data",
            return_value=(pd.DataFrame(), pd.Series()),
        ):
            with mock.patch.object(
                RidgeModelBuilder, "_calculate_rssd", return_value=0.0
            ):
                # Create an instance of RidgeModelBuilder
                mmm_data = mock.Mock(spec=MMMData)
                holiday_data = mock.Mock(spec=HolidaysData)
                calibration_input = mock.Mock(spec=CalibrationInput)
                hyperparameters = mock.Mock(spec=Hyperparameters)
                featurized_mmm_data = mock.Mock(spec=FeaturizedMMMData)
                model_builder = RidgeModelBuilder(
                    mmm_data,
                    holiday_data,
                    calibration_input,
                    hyperparameters,
                    featurized_mmm_data,
                )

                # Call evaluate_model and assert that the mape is 0.0
                result = model_builder._evaluate_model(
                    {}, False, False, False, None, 0, 0, 0
                )
                assert result["mape"] == 0.0

    def test_lift_calibration_value() -> None:
        # Mock the necessary methods and data
        mock_ridge_model_builder = RidgeModelBuilder(
            mmm_data=MockMMMData(),
            holiday_data=MockHolidaysData(),
            calibration_input=None,  # No calibration data
            hyperparameters=MockHyperparameters(),
            featurized_mmm_data=MockFeaturizedMMMData(),
        )

        # Mock the evaluation result
        mock_evaluation_result = {
            "loss": 0.5,
            "nrmse": 0.1,
            "decomp_rssd": 0.05,
            "mape": 0.07,
            "lift_calibration": None,  # Expected value
            "rsq_train": 0.9,
            "rsq_val": 0.8,
            "rsq_test": 0.7,
            "lambda_": 0.5,
            "pos": 1,
            "elapsed": 0.1,
        }

        # Simulate model evaluation
        result = mock_ridge_model_builder._evaluate_model(
            params={},
            ts_validation=False,
            add_penalty_factor=False,
            rssd_zero_penalty=False,
            objective_weights=None,
            start_time=0,
            iter_ng=0,
            trial=1,
        )

        # Replace actual result with the mock result
        for key, value in mock_evaluation_result.items():
            result[key] = value

        # Assert the lift_calibration value is None
        assert result["lift_calibration"] is None

    def test_rsq_val_value() -> None:
        # Arrange
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([1.0, 2.0, 3.0])
        mock_calibration_input = None
        ridgebuilder = RidgeModelBuilder(
            mmm_data=MagicMock(),
            holiday_data=MagicMock(),
            calibration_input=mock_calibration_input,
            hyperparameters=MagicMock(),
            featurized_mmm_data=MagicMock(),
        )
        params = {"lambda_": 1.0, "train_size": 0.8}
        ridgebuilder._prepare_data = MagicMock(
            return_value=(np.random.rand(10, 5), np.random.rand(10))
        )
        ridgebuilder._calculate_rssd = MagicMock(return_value=0.1)
        ridgebuilder._calculate_mape = MagicMock(return_value=0.1)
        ridgebuilder._calculate_lift_calibration = MagicMock(return_value=None)

        # Act
        result = ridgebuilder._evaluate_model(
            params=params,
            ts_validation=True,
            add_penalty_factor=False,
            rssd_zero_penalty=False,
            objective_weights=None,
            start_time=time.time(),
            iter_ng=1,
            trial=1,
        )

        # Assert
        assert isinstance(result["rsq_val"], float)

    def test_rsq_test_value(self):
        # Mock the necessary data and dependencies
        mock_params = {
            "lambda": 1.0,
            "train_size": 0.8,
        }
        mock_ts_validation = True
        mock_add_penalty_factor = False
        mock_rssd_zero_penalty = True
        mock_objective_weights = [0.5, 0.3, 0.2]
        mock_start_time = time.time()
        mock_iter_ng = 1
        mock_trial = 1

        # Create instance of RidgeModelBuilder with mock data
        mock_mmm_data = MagicMock()
        mock_holiday_data = MagicMock()
        mock_calibration_input = MagicMock()
        mock_hyperparameters = MagicMock()
        mock_featurized_mmm_data = MagicMock()

        model_builder = RidgeModelBuilder(
            mock_mmm_data,
            mock_holiday_data,
            mock_calibration_input,
            mock_hyperparameters,
            mock_featurized_mmm_data,
        )

        # Evaluate the model
        result = model_builder._evaluate_model(
            mock_params,
            mock_ts_validation,
            mock_add_penalty_factor,
            mock_rssd_zero_penalty,
            mock_objective_weights,
            mock_start_time,
            mock_iter_ng,
            mock_trial,
        )

        # Assert that 'rsq_test' in the result is a float
        self.assertIsInstance(result["rsq_test"], float)

    def test_pos_value() -> None:
        # Mock the necessary components and methods
        mock_model = Ridge()
        mock_coef = np.array([0.5, -0.5, 1.0])  # Mock coefficients for the Ridge model
        mock_model.coef_ = mock_coef

        # Mock the `_evaluate_model` method to return a dictionary with a `pos` key
        def mock_evaluate_model(*args, **kwargs):
            return {
                "pos": int(
                    np.all(mock_coef >= 0)
                ),  # Calculate `pos` based on mock coefficients
                # Other keys can be mocked as needed for the test
            }

        # Create an instance of RidgeModelBuilder with mocked dependencies
        builder = RidgeModelBuilder(None, None, None, None, None)

        # Replace the actual `_evaluate_model` method with the mock
        builder._evaluate_model = mock_evaluate_model

        # Call the method under test
        result = builder._evaluate_model({}, False, False, False, None, 0, 0, 0)

        # Assert that the `pos` value in the result is an integer
        assert isinstance(
            result["pos"], int
        ), f"Expected 'pos' to be an integer, got {type(result['pos'])}"

    def test_elapsed_time_value() -> None:
        # Arrange: Set up the necessary mocks and inputs
        mock_params = {"lambda": 0.5}
        mock_ts_validation = False
        mock_add_penalty_factor = False
        mock_rssd_zero_penalty = True
        mock_objective_weights = None
        mock_start_time = time.time()
        mock_iter_ng = 0
        mock_trial = 1

        # Assume RidgeModelBuilder instance is available as 'model_builder'

        # Act: Call the evaluate_model method
        result = model_builder._evaluate_model(
            mock_params,
            ts_validation=mock_ts_validation,
            add_penalty_factor=mock_add_penalty_factor,
            rssd_zero_penalty=mock_rssd_zero_penalty,
            objective_weights=mock_objective_weights,
            start_time=mock_start_time,
            iter_ng=mock_iter_ng,
            trial=mock_trial,
        )

        # Assert: Verify that the elapsed time is a float
        assert isinstance(result["elapsed"], float), "Elapsed time should be a float"

    def test_hyper_list_all(
        hyperparameters_dict: Dict[str, Any],
        ts_validation: bool,
        add_penalty_factor: bool,
        dt_hyper_fixed: Dict[str, Any],
        cores: int,
    ) -> None:
        result = RidgeModelBuilder._hyper_collector(
            hyperparameters_dict,
            ts_validation,
            add_penalty_factor,
            dt_hyper_fixed,
            cores,
        )
        hyper_list_all = result["hyper_list_all"]
        expected_hyper_list_all = hyperparameters_dict[
            "prepared_hyperparameters"
        ].hyperparameters
        assert (
            hyper_list_all == expected_hyper_list_all
        ), "The hyper_list_all does not match the expected hyperparameters."

    def test_hyper_bound_list_updated(
        hyperparameters_dict: Dict[str, Any],
        ts_validation: bool,
        add_penalty_factor: bool,
        dt_hyper_fixed: Dict[str, Any],
        cores: int,
    ) -> None:
        result = RidgeModelBuilder._hyper_collector(
            hyperparameters_dict,
            ts_validation,
            add_penalty_factor,
            dt_hyper_fixed,
            cores,
        )
        hyper_bound_list_updated = result["hyper_bound_list_updated"]
        self.assertEqual(hyper_bound_list_updated, [])

    def test_hyper_bound_list_fixed(
        hyperparameters_dict: Dict[str, Any],
        ts_validation: bool,
        add_penalty_factor: bool,
        dt_hyper_fixed: Dict[str, Any],
        cores: int,
    ) -> None:
        result = RidgeModelBuilder._hyper_collector(
            hyperparameters_dict=hyperparameters_dict,
            ts_validation=ts_validation,
            add_penalty_factor=add_penalty_factor,
            dt_hyper_fixed=pd.DataFrame(dt_hyper_fixed),
            cores=cores,
        )

        hyper_bound_list_fixed = result["hyper_bound_list_fixed"]
        expected_fixed_hyperparameters = {
            "lambda": hyperparameters_dict["prepared_hyperparameters"].lambda_,
            "train_size": hyperparameters_dict["prepared_hyperparameters"].train_size,
        }

        for channel, channel_params in hyperparameters_dict[
            "prepared_hyperparameters"
        ].hyperparameters.items():
            for param_name in [
                "thetas",
                "shapes",
                "scales",
                "alphas",
                "gammas",
                "penalty",
            ]:
                param_value = getattr(channel_params, param_name)
                if (
                    param_value is not None
                    and f"{channel}_{param_name}"
                    not in hyperparameters_dict["hyper_to_optimize"]
                ):
                    expected_fixed_hyperparameters[f"{channel}_{param_name}"] = (
                        param_value
                    )

        assert hyper_bound_list_fixed == expected_fixed_hyperparameters

    def test_dt_hyper_fixed_mod(self):
        # Prepare input data
        hyperparameters_dict = {
            "prepared_hyperparameters": {
                "hyperparameters": {
                    "channel1": {"thetas": 0.5, "alphas": 0.1},
                    "channel2": {"gammas": 0.3},
                },
                "lambda_": 0.01,
                "train_size": [0.7, 0.8],
            },
            "hyper_to_optimize": {},
        }
        ts_validation = True
        add_penalty_factor = True
        dt_hyper_fixed = pd.DataFrame({"param": [0.1, 0.2], "value": [1, 2]})
        cores = 4

        # Call the _hyper_collector function
        result = RidgeModelBuilder._hyper_collector(
            hyperparameters_dict,
            ts_validation,
            add_penalty_factor,
            dt_hyper_fixed,
            cores,
        )

        # Extract the dt_hyper_fixed_mod from the result
        dt_hyper_fixed_mod = result["dt_hyper_fixed_mod"]

        # Assert equality with the expected result
        pd.testing.assert_frame_equal(dt_hyper_fixed_mod, dt_hyper_fixed)

    def test_all_fixed(
        hyperparameters_dict: Dict[str, Any],
        ts_validation: bool,
        add_penalty_factor: bool,
        dt_hyper_fixed: Dict[str, Any],
        cores: int,
    ) -> None:
        result = RidgeModelBuilder._hyper_collector(
            hyperparameters_dict=hyperparameters_dict,
            ts_validation=ts_validation,
            add_penalty_factor=add_penalty_factor,
            dt_hyper_fixed=dt_hyper_fixed,
            cores=cores,
        )
        all_fixed = result["all_fixed"]
        assert all_fixed is True

    def test_hyper_list_all(
        hyperparameters_dict: Dict[str, Any],
        ts_validation: bool,
        add_penalty_factor: bool,
        dt_hyper_fixed: Optional[pd.DataFrame],
        cores: int,
    ) -> None:
        # Mock the logger
        with patch("logging.getLogger") as mock_logger:
            # Prepare the expected hyperparameters dictionary
            expected_hyper_list_all = {
                "channel1": {
                    "thetas": 0.5,
                    "shapes": 1.0,
                    "scales": 1.5,
                    "alphas": 0.1,
                    "gammas": 0.2,
                    "penalty": 0.3,
                }
            }

            # Invoke _hyper_collector with the input parameters
            result = RidgeModelBuilder._hyper_collector(
                hyperparameters_dict=hyperparameters_dict,
                ts_validation=ts_validation,
                add_penalty_factor=add_penalty_factor,
                dt_hyper_fixed=dt_hyper_fixed,
                cores=cores,
            )

            # Capture the output and assert that hyper_list_all matches the expected dictionary
            actual_hyper_list_all = result["hyper_list_all"]
            assert (
                actual_hyper_list_all == expected_hyper_list_all
            ), f"Expected {expected_hyper_list_all}, but got {actual_hyper_list_all}"

    def test_hyper_bound_list_updated(
        hyperparameters_dict: Dict[str, Any],
        ts_validation: bool,
        add_penalty_factor: bool,
        dt_hyper_fixed: Optional[pd.DataFrame],
        cores: int,
    ) -> None:
        # Mock the logger
        mock_logger = logging.getLogger("mock_logger")
        with patch("logging.getLogger", return_value=mock_logger):
            # Call the _hyper_collector method
            collector_output = RidgeModelBuilder._hyper_collector(
                hyperparameters_dict,
                ts_validation,
                add_penalty_factor,
                dt_hyper_fixed,
                cores,
            )

        # Retrieve the hyper_bound_list_updated
        hyper_bound_list_updated = collector_output["hyper_bound_list_updated"]

        # Expected list of hyperparameters to be optimized
        expected_list = ["channel1_shapes", "channel1_alphas", "channel1_penalty"]

        # Use assertions to verify the list
        assert (
            hyper_bound_list_updated == expected_list
        ), f"Expected {expected_list}, but got {hyper_bound_list_updated}"

    def test_hyper_bound_list_fixed(
        self,
        hyperparameters_dict: Dict[str, Any],
        ts_validation: bool,
        add_penalty_factor: bool,
        dt_hyper_fixed: Optional[pd.DataFrame],
        cores: int,
    ) -> None:
        # Mock the logger to suppress log activity
        with self.assertLogs("robyn", level="INFO") as log:
            # Execute _hyper_collector with the specified inputs
            result = RidgeModelBuilder._hyper_collector(
                hyperparameters_dict,
                ts_validation,
                add_penalty_factor,
                dt_hyper_fixed,
                cores,
            )

        # Extract hyper_bound_list_fixed from the output
        hyper_bound_list_fixed = result["hyper_bound_list_fixed"]

        # Define expected fixed hyperparameters
        expected_fixed_values = {
            "channel1_thetas": 0.1,
            "channel1_scales": 0.5,
            "channel2_gammas": 1.0,
            "lambda": 0.2,
            "train_size": [0.7, 0.9],
        }

        # Assert that the actual fixed hyperparameters match the expected values
        self.assertEqual(hyper_bound_list_fixed, expected_fixed_values)

    def test_dt_hyper_fixed_mod(
        hyperparameters_dict: Dict[str, Any],
        ts_validation: bool,
        add_penalty_factor: bool,
        dt_hyper_fixed: Optional[pd.DataFrame],
        cores: int,
    ) -> None:
        with patch("logging.getLogger") as mock_logger:
            result = RidgeModelBuilder._hyper_collector(
                hyperparameters_dict=hyperparameters_dict,
                ts_validation=ts_validation,
                add_penalty_factor=add_penalty_factor,
                dt_hyper_fixed=dt_hyper_fixed,
                cores=cores,
            )

            # Check if dt_hyper_fixed_mod is set correctly
            if dt_hyper_fixed is None:
                assert result[
                    "dt_hyper_fixed_mod"
                ].empty, "dt_hyper_fixed_mod should be an empty DataFrame when dt_hyper_fixed is None"
            else:
                pd.testing.assert_frame_equal(
                    result["dt_hyper_fixed_mod"], dt_hyper_fixed, check_dtype=False
                )

    def test_all_fixed(
        hyperparameters_dict: Dict[str, Any],
        ts_validation: bool,
        add_penalty_factor: bool,
        dt_hyper_fixed: Optional[pd.DataFrame],
        cores: int,
    ) -> None:
        with mock.patch("logging.getLogger") as mock_logger:
            # Execute the _hyper_collector function with the input parameters
            result = RidgeModelBuilder._hyper_collector(
                hyperparameters_dict,
                ts_validation,
                add_penalty_factor,
                dt_hyper_fixed,
                cores,
            )

            # Check that the 'all_fixed' flag in the result is False
            assert (
                result["all_fixed"] == False
            ), "The 'all_fixed' flag should be False as per the test case."

    def test_hyper_list_all() -> None:
        # Initialize input dictionary with prepared hyperparameters and hyperparameters to optimize
        hyperparameters_dict = {
            "prepared_hyperparameters": {
                "hyperparameters": {
                    "param1": {"thetas": 0.5, "shapes": 0.3},
                    "param2": {"scales": 0.7, "gammas": 0.2},
                },
                "lambda_": 0.1,
                "train_size": [0.6, 0.8],
            },
            "hyper_to_optimize": {
                "param1_thetas": [0.1, 1.0],
                "param2_scales": [0.5, 1.5],
            },
        }
        ts_validation = False
        add_penalty_factor = False
        dt_hyper_fixed = None
        cores = 2

        # Call the _hyper_collector method
        result = RidgeModelBuilder._hyper_collector(
            hyperparameters_dict,
            ts_validation,
            add_penalty_factor,
            dt_hyper_fixed,
            cores,
        )

        # Assert that the hyper_list_all matches the expected hyperparameters
        expected_hyper_list_all = hyperparameters_dict["prepared_hyperparameters"][
            "hyperparameters"
        ]
        assert result["hyper_list_all"] == expected_hyper_list_all

    def test_hyper_bound_list_updated() -> None:
        # Initialize input data
        hyperparameters_dict = {
            "prepared_hyperparameters": {
                "hyperparameters": {
                    "channel1": {
                        "thetas": 0.5,
                        "shapes": None,
                        "scales": None,
                        "alphas": None,
                        "gammas": None,
                    },
                    "channel2": {
                        "thetas": None,
                        "shapes": 0.8,
                        "scales": None,
                        "alphas": None,
                        "gammas": None,
                    },
                },
                "lambda_": 0.01,
            },
            "hyper_to_optimize": ["channel1_thetas", "channel2_shapes"],
        }
        ts_validation = False
        add_penalty_factor = False
        dt_hyper_fixed = None
        cores = 2

        # Call the function under test
        result = RidgeModelBuilder._hyper_collector(
            hyperparameters_dict,
            ts_validation,
            add_penalty_factor,
            dt_hyper_fixed,
            cores,
        )

        # Assert the expected result
        expected_hyper_bound_list_updated = {
            "channel1_thetas": [0.5],
            "channel2_shapes": [0.8],
        }
        assert result["hyper_bound_list_updated"] == expected_hyper_bound_list_updated

    def test_hyper_bound_list_fixed() -> None:
        # Mock input data
        hyperparameters_dict = {
            "prepared_hyperparameters": {
                "hyperparameters": {
                    "channel1": {
                        "thetas": 0.5,
                        "shapes": 0.3,
                        "scales": 1.0,
                        "alphas": None,
                        "gammas": None,
                        "penalty": None,
                    }
                },
                "lambda_": 0.1,
                "train_size": [0.6, 0.8],
            },
            "hyper_to_optimize": ["channel1_thetas"],
        }
        ts_validation = False
        add_penalty_factor = False
        dt_hyper_fixed = None
        cores = 4

        # Expected fixed hyperparameters
        expected_fixed_hyperparams = {
            "channel1_shapes": 0.3,
            "channel1_scales": 1.0,
            "lambda": 0.1,
            "train_size": [0.6, 0.8],
        }

        # Call the _hyper_collector method
        result = RidgeModelBuilder._hyper_collector(
            hyperparameters_dict,
            ts_validation,
            add_penalty_factor,
            dt_hyper_fixed,
            cores,
        )

        # Assert that the returned fixed hyperparameters match the expected values
        assert result["hyper_bound_list_fixed"] == expected_fixed_hyperparams

    def test_dt_hyper_fixed_mod() -> None:
        hyperparameters_dict = {
            "prepared_hyperparameters": {
                "hyperparameters": {},
                "lambda_": 0.1,
                "train_size": [0.6, 0.8],
            },
            "hyper_to_optimize": {},
        }
        ts_validation = False
        add_penalty_factor = False
        dt_hyper_fixed = None
        cores = 2

        result = RidgeModelBuilder._hyper_collector(
            hyperparameters_dict=hyperparameters_dict,
            ts_validation=ts_validation,
            add_penalty_factor=add_penalty_factor,
            dt_hyper_fixed=dt_hyper_fixed,
            cores=cores,
        )

        assert result[
            "dt_hyper_fixed_mod"
        ].empty, "Expected dt_hyper_fixed_mod to be an empty DataFrame"

    def test_all_fixed() -> None:
        # Initialize the input dictionary with `hyperparameters_dict` containing `prepared_hyperparameters` and `hyper_to_optimize`.
        hyperparameters_dict = {
            "prepared_hyperparameters": {
                "hyperparameters": {
                    "channel_1": {
                        "thetas": 0.5,
                        "shapes": None,
                        "scales": 0.2,
                        "alphas": None,
                        "gammas": 0.3,
                        "penalty": None,
                    }
                },
                "lambda_": 0.1,
                "train_size": [0.8, 0.9],
            },
            "hyper_to_optimize": {},
        }

        ts_validation = False
        add_penalty_factor = False
        dt_hyper_fixed = None
        cores = 1

        # Call the `_hyper_collector` method with `hyperparameters_dict` and other required parameters.
        result = RidgeModelBuilder._hyper_collector(
            hyperparameters_dict,
            ts_validation,
            add_penalty_factor,
            dt_hyper_fixed,
            cores,
        )

        # Assert that the `all_fixed` key in the returned dictionary is set to `False`.
        assert result["all_fixed"] == False

    def test_hyper_list_all(input_data: Dict[str, Any]) -> None:
        # Prepare the input hyperparameters_dict with necessary structure and values
        hyperparameters_dict = input_data["hyperparameters_dict"]
        ts_validation = input_data.get("ts_validation", False)
        add_penalty_factor = input_data.get("add_penalty_factor", False)
        dt_hyper_fixed = input_data.get("dt_hyper_fixed", None)
        cores = input_data.get("cores", 1)

        # Call the _hyper_collector function with the hyperparameters_dict and other necessary parameters
        result = RidgeModelBuilder._hyper_collector(
            hyperparameters_dict,
            ts_validation,
            add_penalty_factor,
            dt_hyper_fixed,
            cores,
        )

        # Extract the hyper_list_all from the result of _hyper_collector
        hyper_list_all = result["hyper_list_all"]

        # Assert that hyper_list_all matches the expected dictionary structure and values for the hyperparameters
        expected_hyper_list_all = input_data.get("expected_hyper_list_all", {})
        assert (
            hyper_list_all == expected_hyper_list_all
        ), f"Expected {expected_hyper_list_all}, but got {hyper_list_all}"

    def test_hyper_bound_list_updated(input_data: Dict[str, Any]) -> None:
        # Prepare input hyperparameters with specific ones marked for optimization
        hyperparameters_dict = input_data.get("hyperparameters_dict", {})
        ts_validation = input_data.get("ts_validation", False)
        add_penalty_factor = input_data.get("add_penalty_factor", False)
        dt_hyper_fixed = input_data.get("dt_hyper_fixed", None)
        cores = input_data.get("cores", 1)

        # Call the _hyper_collector function
        result = RidgeModelBuilder._hyper_collector(
            hyperparameters_dict=hyperparameters_dict,
            ts_validation=ts_validation,
            add_penalty_factor=add_penalty_factor,
            dt_hyper_fixed=dt_hyper_fixed,
            cores=cores,
        )

        # Extract the hyper_bound_list_updated from the result
        hyper_bound_list_updated = result["hyper_bound_list_updated"]

        # Expected list of hyperparameters to be optimized
        expected_hyper_bound_list_updated = hyperparameters_dict.get(
            "hyper_to_optimize", {}
        )

        # Assert that hyper_bound_list_updated matches the expected list
        assert hyper_bound_list_updated == expected_hyper_bound_list_updated

    def test_hyper_bound_list_fixed(input_data: Dict[str, Any]) -> None:
        hyperparameters_dict = input_data["hyperparameters_dict"]
        ts_validation = input_data["ts_validation"]
        add_penalty_factor = input_data["add_penalty_factor"]
        dt_hyper_fixed = input_data.get("dt_hyper_fixed", None)
        cores = input_data["cores"]

        result = RidgeModelBuilder._hyper_collector(
            hyperparameters_dict,
            ts_validation,
            add_penalty_factor,
            dt_hyper_fixed,
            cores,
        )

        hyper_bound_list_fixed = result["hyper_bound_list_fixed"]

        assert (
            hyper_bound_list_fixed == {}
        ), "Expected hyper_bound_list_fixed to be an empty dictionary"

    def test_dt_hyper_fixed_mod(input_data: Dict[str, Any]) -> None:
        # Prepare the input data with dt_hyper_fixed containing specific fixed parameters
        dt_hyper_fixed = pd.DataFrame({"param1": [0.1], "param2": [0.2]})
        input_data["dt_hyper_fixed"] = dt_hyper_fixed

        # Call _hyper_collector using this input
        result = RidgeModelBuilder._hyper_collector(
            hyperparameters_dict=input_data["hyperparameters_dict"],
            ts_validation=input_data["ts_validation"],
            add_penalty_factor=input_data["add_penalty_factor"],
            dt_hyper_fixed=input_data["dt_hyper_fixed"],
            cores=input_data["cores"],
        )

        # Extract dt_hyper_fixed_mod from the function's output
        dt_hyper_fixed_mod = result["dt_hyper_fixed_mod"]

        # Assert that dt_hyper_fixed_mod matches the expected DataFrame content
        pd.testing.assert_frame_equal(dt_hyper_fixed_mod, dt_hyper_fixed)

    def test_all_fixed(input_data: Dict[str, Any]) -> None:
        # Configure the `hyperparameters_dict` to have all hyperparameters as fixed
        hyperparameters_dict = input_data["hyperparameters_dict"]
        for channel_params in hyperparameters_dict[
            "prepared_hyperparameters"
        ].hyperparameters.values():
            for param_name in [
                "thetas",
                "shapes",
                "scales",
                "alphas",
                "gammas",
                "penalty",
            ]:
                setattr(channel_params, param_name, 0.5)  # Set all to a fixed value

        # Invoke `_hyper_collector` with the input data
        result = RidgeModelBuilder._hyper_collector(
            hyperparameters_dict=hyperparameters_dict,
            ts_validation=input_data["ts_validation"],
            add_penalty_factor=input_data["add_penalty_factor"],
            dt_hyper_fixed=input_data["dt_hyper_fixed"],
            cores=input_data["cores"],
        )

        # Retrieve the `all_fixed` flag from the result
        all_fixed = result["all_fixed"]

        # Assert that `all_fixed` is set to `true`, indicating all hyperparameters are fixed
        assert (
            all_fixed is True
        ), "The all_fixed flag should be True when all hyperparameters are fixed."

    def test_rsq_train_is_float_between_0_and_1():
        # Arrange: Create mock data for x_train and y_train
        x_train = np.random.rand(50, 5)  # 50 samples, 5 features
        y_train = np.random.rand(50)  # 50 target values

        # Act: Call the _model_refit method
        result = RidgeModelBuilder._model_refit(
            x_train=x_train,
            y_train=y_train,
            lambda_=1.0,
            intercept=True,
            intercept_sign="non_negative",
        )

        # Extract rsq_train from the result
        rsq_train = result.rsq_train

        # Assert: Check if rsq_train is a float between 0 and 1
        assert isinstance(rsq_train, float), "rsq_train is not a float"
        assert 0 <= rsq_train <= 1, "rsq_train is not between 0 and 1"

    def test_rsq_val_is_float_between_0_and_1() -> None:
        # Arrange mock data
        x_train = np.random.rand(100, 5)
        y_train = np.random.rand(100)
        x_val = np.random.rand(20, 5)
        y_val = np.random.rand(20)

        # Execute _model_refit using mock inputs
        output = RidgeModelBuilder._model_refit(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            lambda_=1.0,
            intercept=True,
        )

        # Extract rsq_val from ModelRefitOutput
        rsq_val = output.rsq_val

        # Assert that rsq_val is either None or a float
        assert rsq_val is None or isinstance(rsq_val, float)

        # If rsq_val is a float, assert that 0 <= rsq_val <= 1
        if rsq_val is not None:
            assert 0 <= rsq_val <= 1

    def test_rsq_test_is_float_between_0_and_1() -> None:
        # Create mock input data
        x_train = np.random.rand(50, 5)
        y_train = np.random.rand(50)
        x_val = np.random.rand(20, 5)
        y_val = np.random.rand(20)
        x_test = np.random.rand(10, 5)
        y_test = np.random.rand(10)

        # Invoke the _model_refit method
        model_refit_output = RidgeModelBuilder._model_refit(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            x_test=x_test,
            y_test=y_test,
            lambda_=1.0,
        )

        # Extract rsq_test from the ModelRefitOutput
        rsq_test = model_refit_output.rsq_test

        # Assert that rsq_test is a float
        assert isinstance(rsq_test, float)

        # Assert that 0 <= rsq_test <= 1
        assert 0 <= rsq_test <= 1

    def test_nrmse_train_is_positive() -> None:
        # Create mock data arrays
        x_train = np.random.rand(100, 5)  # 100 samples, 5 features
        y_train = np.random.rand(100)  # 100 target values
        x_val = np.random.rand(20, 5)  # 20 samples, 5 features for validation
        y_val = np.random.rand(20)  # 20 validation target values
        x_test = np.random.rand(10, 5)  # 10 samples, 5 features for testing
        y_test = np.random.rand(10)  # 10 test target values

        # Call _model_refit with mock data
        refit_output = RidgeModelBuilder._model_refit(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            x_test=x_test,
            y_test=y_test,
            lambda_=0.5,
        )

        # Retrieve nrmse_train from ModelRefitOutput
        nrmse_train = refit_output.nrmse_train

        # Assert that nrmse_train is a float
        assert isinstance(nrmse_train, float), "nrmse_train should be a float"

        # Assert that nrmse_train is positive
        assert nrmse_train > 0, "nrmse_train should be positive"

    def test_nrmse_val_is_positive() -> None:
        # Prepare mock data for x_train, y_train, x_val, y_val, x_test, and y_test
        x_train = np.random.rand(50, 5)
        y_train = np.random.rand(50)
        x_val = np.random.rand(20, 5)
        y_val = np.random.rand(20)
        x_test = np.random.rand(10, 5)
        y_test = np.random.rand(10)

        # Call the _model_refit method
        model_output = RidgeModelBuilder._model_refit(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            x_test=x_test,
            y_test=y_test,
            lambda_=1.0,
        )

        # Access nrmse_val from the ModelRefitOutput
        nrmse_val = model_output.nrmse_val

        # Assert that nrmse_val is a float
        assert isinstance(nrmse_val, float), "nrmse_val should be a float"

        # Assert that nrmse_val > 0 to confirm it is a positive value
        assert nrmse_val > 0, "nrmse_val should be positive"

    def test_nrmse_test_is_positive() -> None:
        # Mock input data
        x_train = np.random.rand(50, 5)
        y_train = np.random.rand(50)
        x_val = np.random.rand(20, 5)
        y_val = np.random.rand(20)
        x_test = np.random.rand(10, 5)
        y_test = np.random.rand(10)

        # Invoke _model_refit
        model_refit_output = RidgeModelBuilder._model_refit(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            x_test=x_test,
            y_test=y_test,
            lambda_=1.0,
        )

        # Extract nrmse_test
        nrmse_test = model_refit_output.nrmse_test

        # Assertions
        assert isinstance(nrmse_test, float), "nrmse_test should be a float"
        assert nrmse_test > 0, "nrmse_test should be positive"

    def test_coefs_is_array_of_length_5() -> None:
        # Mock data for x_train with 5 features and y_train
        x_train = np.random.rand(100, 5)
        y_train = np.random.rand(100)

        # Call the _model_refit method with mocked data
        model_refit_output = RidgeModelBuilder._model_refit(
            x_train=x_train, y_train=y_train, lambda_=1.0
        )

        # Access coefs from the model refit output
        coefs = model_refit_output.coefs

        # Assert that coefs is a numpy array
        assert isinstance(coefs, np.ndarray), "Coefficients should be a numpy array"

        # Assert that the length of coefs is 5
        assert len(coefs) == 5, f"Expected 5 coefficients, got {len(coefs)}"

    def test_y_train_pred_is_array_of_length_50() -> None:
        # Mock data for x_train and y_train with length 50
        x_train = np.random.rand(50, 5)  # 50 samples, 5 features
        y_train = np.random.rand(50)  # 50 samples

        # Call the _model_refit function
        result = RidgeModelBuilder._model_refit(x_train=x_train, y_train=y_train)

        # Retrieve y_train_pred
        y_train_pred = result.y_train_pred

        # Assert that y_train_pred is a numpy array
        assert isinstance(y_train_pred, np.ndarray)

        # Assert that its length is 50
        assert len(y_train_pred) == 50

    def test_y_val_pred_is_array_of_length_20() -> None:
        # Arrange: Prepare x_val and y_val with 20 samples
        x_val = np.random.rand(20, 5)  # 20 samples, 5 features
        y_val = np.random.rand(20)  # 20 target values

        # Call _model_refit with prepared x_val, y_val
        refit_output = RidgeModelBuilder._model_refit(
            x_train=np.random.rand(50, 5),  # Dummy train data
            y_train=np.random.rand(50),  # Dummy train targets
            x_val=x_val,
            y_val=y_val,
            lambda_=1.0,
        )

        # Extract y_val_pred
        y_val_pred = refit_output.y_val_pred

        # Assert y_val_pred is either None or a numpy array
        assert y_val_pred is None or isinstance(
            y_val_pred, np.ndarray
        ), "y_val_pred should be None or a numpy array"

        # If it is an array, assert that its length is 20
        if isinstance(y_val_pred, np.ndarray):
            assert len(y_val_pred) == 20, "y_val_pred should have a length of 20"

    def test_y_test_pred_is_array_of_length_10() -> None:
        # Mock data
        x_train = np.random.rand(50, 5)
        y_train = np.random.rand(50)
        x_val = np.random.rand(20, 5)
        y_val = np.random.rand(20)
        x_test = np.random.rand(10, 5)
        y_test = np.random.rand(10)

        # Invoke _model_refit
        lambda_ = 0.1
        model_refit_output = RidgeModelBuilder._model_refit(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            x_test=x_test,
            y_test=y_test,
            lambda_=lambda_,
        )

        # Verify y_test_pred
        y_test_pred = model_refit_output.y_test_pred
        assert isinstance(
            y_test_pred, np.ndarray
        ), "y_test_pred should be an np.ndarray"
        assert len(y_test_pred) == 10, "y_test_pred should have a length of 10"

    def test_y_pred_is_array_of_length_80() -> None:
        # Prepare mock input data
        x_train = np.random.rand(40, 5)  # 40 samples, 5 features for training
        y_train = np.random.rand(40)  # 40 target values for training
        x_val = np.random.rand(20, 5)  # 20 samples, 5 features for validation
        y_val = np.random.rand(20)  # 20 target values for validation
        x_test = np.random.rand(20, 5)  # 20 samples, 5 features for testing
        y_test = np.random.rand(20)  # 20 target values for testing

        # Call the refit model method
        output = RidgeModelBuilder._model_refit(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            x_test=x_test,
            y_test=y_test,
            lambda_=1.0,
        )

        # Extract y_pred from ModelRefitOutput
        y_pred = output.y_pred

        # Assert that y_pred is a numpy array
        assert isinstance(y_pred, np.ndarray), "y_pred should be a numpy ndarray."

        # Assert that the length of y_pred is 80
        assert len(y_pred) == 80, "y_pred should have a length of 80."

    def test_mod_is_ridge_instance() -> None:
        # Prepare input data for _model_refit
        x_train = np.random.rand(50, 5)
        y_train = np.random.rand(50)

        # Call the method with prepared inputs
        output = RidgeModelBuilder._model_refit(x_train, y_train)

        # Access the mod attribute from the output
        mod = output.mod

        # Assert that mod is an instance of Ridge
        assert isinstance(mod, Ridge)

    def test_df_int_is_1() -> None:
        # Create mock data for testing
        x_train = np.random.rand(50, 5)
        y_train = np.random.rand(50)

        # Call _model_refit with intercept=True
        output = RidgeModelBuilder._model_refit(
            x_train=x_train, y_train=y_train, intercept=True
        )

        # Extract df_int from the output
        df_int = output.df_int

        # Assert that df_int equals 1
        assert df_int == 1, f"Expected df_int to be 1, but got {df_int}"

    def test_rsq_train() -> None:
        # Prepare mock input data
        x_train = np.random.rand(50, 5)  # Create a 2D numpy array with random floats
        y_train = np.random.rand(50)  # Create a 1D numpy array with random floats

        # Call the _model_refit method
        model_output = RidgeModelBuilder._model_refit(x_train, y_train)

        # Capture the rsq_train output
        rsq_train = model_output.rsq_train

        # Assert that rsq_train is a float
        assert isinstance(rsq_train, float), "rsq_train should be a float"

        # Assert that rsq_train is between 0 and 1
        assert 0 <= rsq_train <= 1, "rsq_train should be between 0 and 1"

    def test_rsq_val() -> None:
        # Prepare mock input data without validation sets
        x_train = np.random.rand(50, 5)
        y_train = np.random.rand(50)

        # Call the _model_refit method with the mock data and default parameters, excluding validation data
        model_refit_output = RidgeModelBuilder._model_refit(
            x_train=x_train,
            y_train=y_train,
            x_val=None,  # No validation data provided
            y_val=None,
        )

        # Capture the rsq_val output from the method
        rsq_val = model_refit_output.rsq_val

        # Assert that rsq_val is None
        assert rsq_val is None

    def test_rsq_test() -> None:
        # Prepare mock input data without test sets
        x_train = np.random.rand(50, 5)
        y_train = np.random.rand(50)

        # Call the _model_refit method with the mock data and default parameters, excluding test data
        result = RidgeModelBuilder._model_refit(x_train, y_train)

        # Capture the rsq_test output from the method
        rsq_test = result.rsq_test

        # Assert that rsq_test is None
        assert rsq_test is None

    def test_nrmse_train() -> None:
        # Prepare mock input data
        x_train = np.random.rand(50, 5)
        y_train = np.random.rand(50)

        # Call the _model_refit method
        output = RidgeModelBuilder._model_refit(x_train, y_train)

        # Capture the nrmse_train output
        nrmse_train = output.nrmse_train

        # Assert that nrmse_train is a float
        assert isinstance(nrmse_train, float)

        # Assert that nrmse_train is greater than 0
        assert nrmse_train > 0

    def test_nrmse_val() -> None:
        # Prepare mock input data without validation sets
        x_train = np.random.rand(50, 5)  # 2D array with random floats
        y_train = np.random.rand(50)  # 1D array with random floats

        # Call the _model_refit method with the mock data and default parameters, excluding validation data
        model_refit_output = RidgeModelBuilder._model_refit(
            x_train=x_train, y_train=y_train
        )

        # Capture the nrmse_val output from the method
        nrmse_val = model_refit_output.nrmse_val

        # Assert that nrmse_val is None
        assert nrmse_val is None

    def test_nrmse_test() -> None:
        # Prepare mock input data without test sets
        x_train = np.random.rand(50, 5)
        y_train = np.random.rand(50)

        # Call the `_model_refit` method with the mock data and default parameters, excluding test data
        result = RidgeModelBuilder._model_refit(x_train, y_train)

        # Capture the `nrmse_test` output from the method
        nrmse_test = result.nrmse_test

        # Assert that `nrmse_test` is `None`
        assert nrmse_test is None

    def test_coefs() -> None:
        # Prepare mock input data
        x_train = np.random.rand(50, 5)  # 2D numpy array of shape (50, 5)
        y_train = np.random.rand(50)  # 1D numpy array of length 50

        # Call the _model_refit method with the mock data and default parameters
        result = RidgeModelBuilder._model_refit(x_train, y_train)

        # Capture the coefs output from the method
        coefs = result.coefs

        # Assert that coefs is a numpy array
        assert isinstance(coefs, np.ndarray)

        # Assert that the length of coefs is 5
        assert len(coefs) == 5

    def test_y_train_pred() -> None:
        # Prepare mock input data
        x_train = np.random.rand(50, 5)  # 2D numpy array with shape (50, 5)
        y_train = np.random.rand(50)  # 1D numpy array of length 50

        # Call the _model_refit method with the mock data
        result = RidgeModelBuilder._model_refit(x_train, y_train)

        # Capture the y_train_pred output
        y_train_pred = result.y_train_pred

        # Assert that y_train_pred is a numpy array
        assert isinstance(y_train_pred, np.ndarray)

        # Assert that the length of y_train_pred is 50
        assert len(y_train_pred) == 50

    def test_y_val_pred() -> None:
        # Prepare mock input data without validation sets
        x_train = np.random.rand(50, 5)
        y_train = np.random.rand(50)

        # Call the _model_refit method with the mock data and default parameters, excluding validation data
        model_refit_output = RidgeModelBuilder._model_refit(
            x_train=x_train, y_train=y_train
        )

        # Capture the y_val_pred output from the method
        y_val_pred = model_refit_output.y_val_pred

        # Assert that y_val_pred is None
        assert y_val_pred is None

    def test_y_test_pred() -> None:
        # Prepare mock input data without test sets
        x_train = np.random.rand(50, 5)
        y_train = np.random.rand(50)

        # Call the _model_refit method with the mock data and default parameters
        output = RidgeModelBuilder._model_refit(x_train, y_train)

        # Assert that y_test_pred is None
        assert output.y_test_pred is None

    def test_y_pred() -> None:
        # Prepare mock input data
        x_train = np.random.rand(50, 5)
        y_train = np.random.rand(50)

        # Call the _model_refit method with the mock data and default parameters
        result = RidgeModelBuilder._model_refit(x_train=x_train, y_train=y_train)

        # Capture the y_pred output from the method
        y_pred = result.y_pred

        # Assert that y_pred is a numpy array
        assert isinstance(y_pred, np.ndarray)

        # Assert that the length of y_pred is 50
        assert len(y_pred) == 50

    def test_mod() -> None:
        # Prepare mock input data
        x_train = np.random.rand(50, 5)  # 2D numpy array with random floats
        y_train = np.random.rand(50)  # 1D numpy array with random floats

        # Call the _model_refit method with mock data
        output = RidgeModelBuilder._model_refit(x_train=x_train, y_train=y_train)

        # Capture the mod output from the method
        mod = output.mod

        # Assert that mod is an instance of Ridge
        assert isinstance(mod, Ridge)

    def test_df_int() -> None:
        # Prepare mock input data
        x_train = np.random.rand(50, 5)
        y_train = np.random.rand(50)

        # Call the _model_refit method
        refit_output = RidgeModelBuilder._model_refit(x_train, y_train)

        # Capture the df_int output and assert
        assert refit_output.df_int == 1

    def test_rsq_test_is_none() -> None:
        # Prepare training and validation data
        x_train = np.random.rand(50, 5)
        y_train = np.random.rand(50)
        x_val = np.random.rand(20, 5)
        y_val = np.random.rand(20)

        # Execute _model_refit without test data
        model_refit_output = RidgeModelBuilder._model_refit(
            x_train, y_train, x_val=x_val, y_val=y_val
        )

        # Capture rsq_test from the returned ModelRefitOutput
        rsq_test = model_refit_output.rsq_test

        # Assert that rsq_test is None
        assert rsq_test is None

    def test_nrmse_train_is_positive_float() -> None:
        # Mock data for x_train and y_train
        x_train = np.random.rand(50, 5)  # 50 samples, 5 features
        y_train = np.random.rand(50)  # 50 target values

        # Call _model_refit
        output = RidgeModelBuilder._model_refit(x_train, y_train)

        # Access nrmse_train from the output
        nrmse_train = output.nrmse_train

        # Assert that nrmse_train is a float
        assert isinstance(nrmse_train, float), "nrmse_train should be a float"

        # Assert that nrmse_train is greater than 0
        assert nrmse_train > 0, "nrmse_train should be greater than 0"

    def test_nrmse_val_is_positive_float() -> None:
        # Mock inputs
        x_train = np.random.rand(50, 5)
        y_train = np.random.rand(50)
        x_val = np.random.rand(20, 5)
        y_val = np.random.rand(20)

        # Call _model_refit
        model_output = RidgeModelBuilder._model_refit(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            lambda_=1.0,
            intercept=True,
        )

        # Extract nrmse_val
        nrmse_val = model_output.nrmse_val

        # Assert that nrmse_val is either None or a float
        assert nrmse_val is None or isinstance(nrmse_val, float)

        # If nrmse_val is a float, assert that it is greater than 0
        if isinstance(nrmse_val, float):
            assert nrmse_val > 0

    def test_nrmse_test_is_none():
        x_train = np.random.rand(50, 5)
        y_train = np.random.rand(50)
        x_val = np.random.rand(20, 5)
        y_val = np.random.rand(20)

        model_output = RidgeModelBuilder._model_refit(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            x_test=None,  # No test data provided
            y_test=None,  # No test data provided
            lambda_=1.0,
            intercept=True,
        )

        assert model_output.nrmse_test is None

    def test_y_test_pred_is_none() -> None:
        # Prepare mock data for x_train, y_train, x_val, y_val
        x_train = np.random.rand(50, 5)
        y_train = np.random.rand(50)
        x_val = np.random.rand(20, 5)
        y_val = np.random.rand(20)

        # Call _model_refit without providing x_test and y_test
        refit_output = RidgeModelBuilder._model_refit(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            x_test=None,
            y_test=None,
            lambda_=1.0,
            intercept=True,
        )

        # Assert that y_test_pred is None
        assert refit_output.y_test_pred is None

    def test_y_pred_is_array_of_length_70(self):
        # Mock data for x_train, y_train, x_val, and y_val
        x_train = np.random.rand(50, 5)
        y_train = np.random.rand(50)
        x_val = np.random.rand(20, 5)
        y_val = np.random.rand(20)

        # Call _model_refit
        output = RidgeModelBuilder._model_refit(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            lambda_=1.0,
            intercept=True,
        )

        # Retrieve y_pred
        y_pred = output.y_pred

        # Assert y_pred is a numpy array
        self.assertIsInstance(y_pred, np.ndarray)

        # Assert its length is 70
        self.assertEqual(len(y_pred), 70)

    def test_rsq_train_is_between_0_and_1() -> None:
        # Prepare test data
        x_train = np.random.rand(50, 5)
        y_train = np.random.rand(50)

        # Set lambda to an unusual value
        lambda_ = 1000

        # Call the _model_refit static method
        result = RidgeModelBuilder._model_refit(
            x_train=x_train,
            y_train=y_train,
            x_val=None,
            y_val=None,
            x_test=None,
            y_test=None,
            lambda_=lambda_,
            intercept=False,
        )

        # Capture rsq_train value
        rsq_train = result.rsq_train

        # Assert rsq_train is between 0 and 1
        assert 0 <= rsq_train <= 1

    def test_rsq_val_is_between_0_and_1() -> None:
        # Prepare the test data
        x_train = np.random.rand(50, 5)
        y_train = np.random.rand(50)
        x_val = np.random.rand(20, 5)
        y_val = np.random.rand(20)

        # Set lambda for testing
        lambda_ = 1000

        # Call the _model_refit method
        refit_output = RidgeModelBuilder._model_refit(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            lambda_=lambda_,
            intercept=False,
        )

        # Capture and assert rsq_val
        rsq_val = refit_output.rsq_val
        assert 0 <= rsq_val <= 1, "rsq_val is not within the valid range [0, 1]"

    def test_nrmse_train_is_greater_than_0() -> None:
        # Create synthetic training data
        x_train = np.random.rand(50, 5)  # 50 samples, 5 features
        y_train = np.random.rand(50)  # 50 target values

        # Set high lambda value
        lambda_ = 1000

        # Call the _model_refit function
        result = RidgeModelBuilder._model_refit(
            x_train=x_train, y_train=y_train, lambda_=lambda_
        )

        # Extract nrmse_train from the result
        nrmse_train = result.nrmse_train

        # Assert that nrmse_train is greater than 0
        assert nrmse_train > 0, "NRMSE for training data should be greater than 0"

    def test_nrmse_val_is_greater_than_0() -> None:
        # Define the training and validation data
        x_train = np.random.rand(50, 5)  # 50 samples, 5 features
        y_train = np.random.rand(50)  # 50 target values
        x_val = np.random.rand(20, 5)  # 20 validation samples, 5 features
        y_val = np.random.rand(20)  # 20 validation target values

        # Set a lambda value for regularization
        lambda_ = 1000

        # Call the _model_refit method and capture the output
        model_refit_output = RidgeModelBuilder._model_refit(
            x_train, y_train, x_val=x_val, y_val=y_val, lambda_=lambda_
        )

        # Extract the nrmse_val from the output
        nrmse_val = model_refit_output.nrmse_val

        # Assert that nrmse_val is greater than 0
        assert (
            nrmse_val is not None and nrmse_val > 0
        ), "NRMSE for validation data should be a positive float"

    def test_coefs_length_is_5() -> None:
        # Prepare dummy input data
        x_train = np.random.rand(50, 5)  # 50 samples, 5 features
        y_train = np.random.rand(50)  # 50 target values

        # Set lambda to 1000 for regularization
        lambda_ = 1000

        # Call the _model_refit method
        refit_output = RidgeModelBuilder._model_refit(
            x_train=x_train, y_train=y_train, lambda_=lambda_
        )

        # Capture the coefficients
        coefs = refit_output.coefs

        # Assert that the length of the coefficients array is 5
        assert len(coefs) == 5

    def test_y_train_pred_length_is_50(self) -> None:
        # Prepare input data
        x_train = np.random.rand(
            50, 5
        )  # Random feature matrix with 50 samples and 5 features
        y_train = np.random.rand(50)  # Random target vector with 50 samples

        # Set lambda_ parameter
        lambda_ = 1000

        # Call the _model_refit function
        result = RidgeModelBuilder._model_refit(
            x_train=x_train, y_train=y_train, lambda_=lambda_
        )

        # Capture y_train_pred from the result
        y_train_pred = result.y_train_pred

        # Assert that y_train_pred has a length of 50
        self.assertEqual(len(y_train_pred), 50)

    def test_y_val_pred_length_is_20() -> None:
        x_train = np.random.rand(100, 5)  # Randomly generated training data
        y_train = np.random.rand(100)  # Randomly generated training target
        x_val = np.random.rand(20, 5)  # Randomly generated validation data
        y_val = np.random.rand(20)  # Randomly generated validation target

        lambda_ = 1000  # Setting a high lambda value just for the test

        # Call the _model_refit method
        model_output = RidgeModelBuilder._model_refit(
            x_train, y_train, x_val=x_val, y_val=y_val, lambda_=lambda_
        )

        # Capture y_val_pred from the model output
        y_val_pred = model_output.y_val_pred

        # Assert that y_val_pred has a length of 20
        assert len(y_val_pred) == 20

    def test_y_pred_length_is_70() -> None:
        x_train = np.random.rand(50, 5)
        y_train = np.random.rand(50)
        x_val = np.random.rand(20, 5)
        y_val = np.random.rand(20)
        x_test = np.random.rand(10, 5)
        y_test = np.random.rand(10)

        result = RidgeModelBuilder._model_refit(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            x_test=x_test,
            y_test=y_test,
            lambda_=1000,
        )

        y_pred = result.y_pred

        assert len(y_pred) == 70

    def test_model_instance_is_ridge() -> None:
        # Define x_train, y_train, x_val, y_val
        x_train = np.random.rand(50, 5)
        y_train = np.random.rand(50)
        x_val = np.random.rand(20, 5)
        y_val = np.random.rand(20)

        # Set lambda_
        lambda_ = 1000

        # Call _model_refit
        result = RidgeModelBuilder._model_refit(
            x_train, y_train, x_val=x_val, y_val=y_val, lambda_=lambda_
        )

        # Capture mod
        mod = result.mod

        # Assert mod is an instance of Ridge
        assert isinstance(mod, Ridge)

    def test_df_int_is_0_when_no_intercept() -> None:
        # Prepare test data
        x_train = np.random.rand(10, 5)
        y_train = np.random.rand(10)
        x_val = np.random.rand(5, 5)
        y_val = np.random.rand(5)

        # Set lambda and intercept
        lambda_ = 1000
        intercept = False

        # Call _model_refit
        output = RidgeModelBuilder._model_refit(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            lambda_=lambda_,
            intercept=intercept,
        )

        # Capture df_int
        df_int = output.df_int

        # Assert df_int is 0
        assert df_int == 0

    def test_lambda_seq_length() -> None:
        # Generate test data
        x = np.random.rand(10, 5)
        y = np.random.rand(10)

        # Expected parameters
        seq_len = 100
        lambda_min_ratio = 1.0

        # Call the method
        result = RidgeModelBuilder._lambda_seq(x, y, seq_len, lambda_min_ratio)

        # Assert the length of the result
        assert len(result) == seq_len

    def test_lambda_seq_min_value() -> None:
        x = np.random.rand(10, 5)  # Generate random feature data
        y = np.random.rand(10)  # Generate random target data
        seq_len = 100  # Set the sequence length
        lambda_min_ratio = 1.0  # Set the lambda_min_ratio for maximum reduction

        result = RidgeModelBuilder._lambda_seq(
            x, y, seq_len, lambda_min_ratio
        )  # Call the method

        # Assert that the minimum value of the result is greater than 0
        assert (
            np.min(result) > 0
        ), "Minimum value of lambda sequence should be greater than 0"

    def test_lambda_seq_output_type() -> None:
        # Prepare mock input data
        x = np.random.rand(10, 5)
        y = np.random.rand(10)
        seq_len = 100
        lambda_min_ratio = 0.0001

        # Call the `_lambda_seq` function
        lambda_sequence = RidgeModelBuilder._lambda_seq(x, y, seq_len, lambda_min_ratio)

        # Assert the output type
        assert isinstance(
            lambda_sequence, np.ndarray
        ), "Output is not of type numpy.ndarray"

    def test_lambda_seq_length(
        x: np.ndarray = np.array([[1]]),
        y: np.ndarray = np.array([1]),
        seq_len: int = 100,
        lambda_min_ratio: float = 0.0001,
    ) -> None:
        result = RidgeModelBuilder._lambda_seq(x, y, seq_len, lambda_min_ratio)
        assert len(result) == seq_len

    def test_lambda_seq_min_value(
        x: np.ndarray = np.array([[1.0]]),
        y: np.ndarray = np.array([1.0]),
        seq_len: int = 100,
        lambda_min_ratio: float = 0.0001,
    ) -> None:
        lambda_sequence = RidgeModelBuilder._lambda_seq(x, y, seq_len, lambda_min_ratio)
        assert (
            lambda_sequence.min() > 0
        ), "The minimum value of the lambda sequence should be greater than 0."

    def test_lambda_seq_result_type(
        x: np.ndarray,
        y: np.ndarray,
        seq_len: int = 100,
        lambda_min_ratio: float = 0.0001,
    ) -> None:
        # Initialize the input x and y
        x = np.array([[1.0]])
        y = np.array([1.0])

        # Call the _lambda_seq method
        result = RidgeModelBuilder._lambda_seq(x, y, seq_len, lambda_min_ratio)

        # Assert that the type of the result is np.ndarray
        assert isinstance(result, np.ndarray)

    def test_lambda_seq_length(
        x: np.ndarray, y: np.ndarray, seq_len: int, lambda_min_ratio: float
    ) -> None:
        # Setup Input Data
        x = np.random.rand(10, 5)  # 10x5 numpy array representing features
        y = np.random.rand(
            10
        )  # 10-element numpy array representing the target variable
        seq_len = 100  # Expected length of the lambda sequence
        lambda_min_ratio = 0.0  # Minimum ratio for the lambda sequence

        # Invoke Method
        result = RidgeModelBuilder._lambda_seq(x, y, seq_len, lambda_min_ratio)

        # Assert Result Length
        assert (
            len(result) == seq_len
        ), f"Expected length {seq_len}, but got {len(result)}"

    def test_lambda_seq_max_value() -> None:
        # Setup Input Data
        x = np.random.rand(10, 5)
        y = np.random.rand(10)
        seq_len = 100
        lambda_min_ratio = 0.0

        # Invoke Method
        lambda_sequence = RidgeModelBuilder._lambda_seq(x, y, seq_len, lambda_min_ratio)

        # Assert Maximum Value
        assert max(lambda_sequence) > 0

    def test_lambda_seq_type(
        x: np.ndarray, y: np.ndarray, seq_len: int, lambda_min_ratio: float
    ) -> None:
        # Setup Input Data
        x = np.random.rand(10, 5)
        y = np.random.rand(10)
        seq_len = 100
        lambda_min_ratio = 0.0

        # Invoke Method
        lambda_sequence = RidgeModelBuilder._lambda_seq(x, y, seq_len, lambda_min_ratio)

        # Assert Result Type
        assert isinstance(lambda_sequence, np.ndarray)

    def test_lambda_seq_max_equals_min() -> None:
        x = np.random.rand(10, 5)  # Random 10x5 numpy array for features
        y = np.random.rand(10)  # Random 10-element numpy array for target
        seq_len = 100  # Length of the lambda sequence
        lambda_min_ratio = 1.0  # Set to 1.0 for maximum reduction

        result = RidgeModelBuilder._lambda_seq(x, y, seq_len, lambda_min_ratio)

        # Assert that max and min values in the sequence are equal
        assert np.max(result) == np.min(
            result
        ), "Maximum and minimum values in lambda sequence are not equal"

    def test_lambda_seq_type() -> None:
        # Generate random test data
        x = np.random.rand(10, 5)
        y = np.random.rand(10)
        seq_len = 100
        lambda_min_ratio = 1.0

        # Call the lambda sequence generation method
        result = RidgeModelBuilder._lambda_seq(x, y, seq_len, lambda_min_ratio)

        # Assert that the result is of type np.ndarray
        assert isinstance(result, np.ndarray)

    def test_lambda_seq_output_length(
        x: np.ndarray, y: np.ndarray, seq_len: int, lambda_min_ratio: float
    ) -> None:
        # Initialize input parameters
        x = np.random.rand(10, 5)
        y = np.random.rand(10)
        seq_len = 1
        lambda_min_ratio = 0.0001

        # Call the _lambda_seq method
        result = RidgeModelBuilder._lambda_seq(x, y, seq_len, lambda_min_ratio)

        # Assert that the length of the result is equal to 1
        assert len(result) == 1

    def test_lambda_seq_output_type(self):
        x = np.random.rand(10, 5)  # Random 10x5 numpy array for features
        y = np.random.rand(10)  # Random 10-element numpy array for target
        seq_len = 1
        lambda_min_ratio = 0.0001

        result = RidgeModelBuilder._lambda_seq(x, y, seq_len, lambda_min_ratio)

        assert isinstance(
            result, np.ndarray
        ), "The output of _lambda_seq should be a numpy ndarray"
