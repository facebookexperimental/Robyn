import unittest
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
from robyn.modeling.pareto.pareto_optimizer import (
    ParetoOptimizer,
    ParetoResult,
    ParetoData,
)  # Adjust the import path as needed


class TestParetoOptimizer(unittest.TestCase):

    def setUp(self):
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
        self.mmm_data = MMMData(data=data, mmmdata_spec=mmmdata_spec)
        # self.mmm_data = MagicMock(spec=MMMData)
        self.model_outputs = MagicMock(spec=ModelOutputs)
        self.hyper_parameter = MagicMock(spec=Hyperparameters)
        self.featurized_mmm_data = type('', (), {})()
        self.featurized_mmm_data.dt_modRollWind = pd.DataFrame({
            'ds': pd.date_range(start='2023-01-01', periods=10, freq='D')
        })
        self.featurized_mmm_data.dt_mod = pd.DataFrame() 
        self.holidays_data = MagicMock(spec=HolidaysData)

        # Instance of ParetoOptimizer
        self.optimizer = ParetoOptimizer(
            mmm_data=self.mmm_data,
            model_outputs=self.model_outputs,
            hyper_parameter=self.hyper_parameter,
            featurized_mmm_data=self.featurized_mmm_data,
            holidays_data=self.holidays_data,
        )

    def test_optimize(self):
        # Setup mock return values for methods called within optimize
        self.optimizer._aggregate_model_data = MagicMock(
            return_value={
                "result_hyp_param": pd.DataFrame(),
                "result_calibration": pd.DataFrame(),
            }
        )
        self.optimizer._compute_pareto_fronts = MagicMock(return_value=pd.DataFrame())
        self.optimizer.prepare_pareto_data = MagicMock(
            return_value=ParetoData(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), [])
        )
        self.optimizer._compute_response_curves = MagicMock(
            return_value=ParetoData(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), [])
        )
        self.optimizer._generate_plot_data = MagicMock(
            return_value={
                "pareto_solutions": [],
                "mediaVecCollect": pd.DataFrame(),
                "xDecompVecCollect": pd.DataFrame(),
                "plotDataCollect": {},
                "df_caov_pct_all": pd.DataFrame(),
            }
        )

        # Run the optimize function
        result = self.optimizer.optimize()

        # Assertions to check function calls and return values
        self.optimizer._aggregate_model_data.assert_called_once()
        self.optimizer._compute_pareto_fronts.assert_called_once()
        self.optimizer.prepare_pareto_data.assert_called_once()
        self.optimizer._compute_response_curves.assert_called_once()
        self.optimizer._generate_plot_data.assert_called_once()

        self.assertIsInstance(result, ParetoResult)

    def test_aggregate_model_data(self):
        # Setup mock return values
        self.model_outputs.hyper_fixed = True
        self.model_outputs.trials = [
            MagicMock(
                result_hyp_param=pd.DataFrame(), 
                x_decomp_agg=pd.DataFrame({'solID': [1]}),
            )
        ]

        # Run the _aggregate_model_data function
        result = self.optimizer._aggregate_model_data(calibrated=False)

        # Assertions to check the return value
        self.assertIsInstance(result, dict)
        self.assertIn("result_hyp_param", result)
        self.assertIn("x_decomp_agg", result)

    @patch('robyn.modeling.pareto.pareto_optimizer.ParetoOptimizer._pareto_fronts')
    def test_compute_pareto_fronts_hyper_fixed_false(self, mock_pareto_fronts):
        # Setup mock data
        mock_pareto_fronts.return_value = pd.DataFrame({
            'x': [],  # Corresponds to 'nrmse'
            'y': [],  # Corresponds to 'decomp.rssd'
            'pareto_front': []
        })
        aggregated_data = {
            "result_hyp_param": pd.DataFrame({
                "mape": [],  # Include the 'mape' column
                "nrmse": [],  # Include the 'nrmse' column
                "decomp.rssd": [],  # Include the 'decomp.rssd' column
                "nrmse_train": [],
                "solID": [],
                "iterNG": [],
                "iterPar": []
            }),
            "x_decomp_agg": pd.DataFrame({
                "rn": [],  # Include the 'rn' column
                "solID": [],
                "coef": []
            }),
            "result_calibration": None,
        }

        self.model_outputs.hyper_fixed = False 
        self.model_outputs.ts_validation = None

        # Run the _compute_pareto_fronts function
        result = self.optimizer._compute_pareto_fronts(
            aggregated_data=aggregated_data,
            pareto_fronts="auto",
            min_candidates=100,
            calibration_constraint=0.1,
        )

        # Assertions to check the return value
        self.assertIsInstance(result, pd.DataFrame)

    @patch('robyn.modeling.pareto.pareto_optimizer.ParetoOptimizer._pareto_fronts')
    def test_compute_pareto_fronts_hyper_fixed_true(self, mock_pareto_fronts):
        # Setup mock data
        mock_pareto_fronts.return_value = pd.DataFrame({
            'x': [],  # Corresponds to 'nrmse'
            'y': [],  # Corresponds to 'decomp.rssd'
            'pareto_front': []
        })
        aggregated_data = {
            "result_hyp_param": pd.DataFrame({
                "mape": [],  # Include the 'mape' column
                "nrmse": [],  # Include the 'nrmse' column
                "decomp.rssd": [],  # Include the 'decomp.rssd' column
                "nrmse_train": [],
                "solID": [],
                "iterNG": [],
                "iterPar": []
            }),
            "x_decomp_agg": pd.DataFrame({
                "rn": [],  # Include the 'rn' column
                "solID": [],
                "coef": []
            }),
            "result_calibration": None,
        }

        self.model_outputs.hyper_fixed = True 
        self.model_outputs.ts_validation = None

        # Run the _compute_pareto_fronts function
        result = self.optimizer._compute_pareto_fronts(
            aggregated_data=aggregated_data,
            pareto_fronts="auto",
            min_candidates=100,
            calibration_constraint=0.1,
        )

        # Assertions to check the return value
        self.assertIsInstance(result, pd.DataFrame)

    def test_prepare_pareto_data_hyper_fixed_true(self):
        # Setup mock data
        aggregated_data = {
            "result_hyp_param": pd.DataFrame({
                "robynPareto": [1, 2, 3],
                "solID": ["sol1", "sol2", "sol3"]
            }),
            "x_decomp_agg": pd.DataFrame({
                "solID": ["sol1", "sol2", "sol3"],
                "some_other_column": [10, 20, 30]
            }),
            "result_calibration": None,
        }

        trial_mock = MagicMock()
        trial_mock.decomp_spend_dist = pd.DataFrame({
            "trial": [1, 2, 3],
            "iterNG": [1, 1, 1],
            "iterPar": [1, 2, 3],
            "solID": ["sol1", "sol2", "sol3"]
        })
        self.model_outputs.trials = [trial_mock]
        self.model_outputs.hyper_fixed = True 

        # Run the prepare_pareto_data function
        result = self.optimizer.prepare_pareto_data(
            aggregated_data=aggregated_data,
            pareto_fronts="auto",
            min_candidates=100,
            calibrated=False,
        )

        # Assertions to check the return value
        self.assertIsInstance(result, ParetoData)

    def test_prepare_pareto_data_hyper_fixed_false(self):
        # Setup mock data
        aggregated_data = {
            "result_hyp_param": pd.DataFrame({
                "robynPareto": [1, 2, 3],
                "solID": ["sol1", "sol2", "sol3"]
            }),
            "x_decomp_agg": pd.DataFrame({
                "solID": ["sol1", "sol2", "sol3"],
                "some_other_column": [10, 20, 30]
            }),
            "result_calibration": None,
        }

        trial_mock = MagicMock()
        trial_mock.decomp_spend_dist = pd.DataFrame({
            "trial": [1, 2, 3],
            "iterNG": [1, 1, 1],
            "iterPar": [1, 2, 3],
            "solID": ["sol1", "sol2", "sol3"]
        })
        self.model_outputs.trials = [trial_mock]
        self.model_outputs.hyper_fixed = False 

        # Run the prepare_pareto_data function
        result = self.optimizer.prepare_pareto_data(
            aggregated_data=aggregated_data,
            pareto_fronts="auto",
            min_candidates=2,
            calibrated=False,
        )

        # Assertions to check the return value
        self.assertIsInstance(result, ParetoData)

    def test_run_dt_resp(self):
        # Setup mock data
        row = pd.Series({"solID": "test", "rn": "media"})
        pareto_data = ParetoData(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), [])

        # Run the run_dt_resp function
        result = self.optimizer.run_dt_resp(row=row, paretoData=pareto_data)

        # Assertions to check the return value
        self.assertIsInstance(result, pd.Series)

    def test_compute_response_curves(self):
        # Setup mock data
        decomp_spend_dist_df = pd.DataFrame({
            'solID': [1, 2],  # Example solution IDs
            'rn': [1, 2],     # Example row numbers
            'mean_response': [100, 200],
            'mean_spend': [50, 100],
            'xDecompAgg': [150, 300],
            'total_spend': [500, 1000],
            # Add other necessary columns with mock data
        })
        pareto_data = ParetoData(decomp_spend_dist_df, pd.DataFrame(), pd.DataFrame(), [])
        aggregated_data = {
            "result_hyp_param": pd.DataFrame(),
            "x_decomp_agg": pd.DataFrame({
                'solID': [1, 2],
                'rn': [1, 2],
                # Add other necessary columns with mock data
            }),
            "result_calibration": None,
        }

        self.model_outputs.cores = 4

        # Run the _compute_response_curves function
        result = self.optimizer._compute_response_curves(
            pareto_data=pareto_data, aggregated_data=aggregated_data
        )

        # Assertions to check the return value
        self.assertIsInstance(result, ParetoData)

    def test_generate_plot_data(self):
        # Setup mock data
        pareto_data = ParetoData(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), [])
        aggregated_data = {
            "result_hyp_param": pd.DataFrame(),
            "x_decomp_agg": pd.DataFrame(),
            "result_calibration": None,
        }

        # Run the _generate_plot_data function
        result = self.optimizer._generate_plot_data(
            aggregated_data=aggregated_data, pareto_data=pareto_data
        )

        # Assertions to check the return value
        self.assertIsInstance(result, dict)
        self.assertIn("pareto_solutions", result)
        self.assertIn("mediaVecCollect", result)
        self.assertIn("xDecompVecCollect", result)
        self.assertIn("plotDataCollect", result)
        self.assertIn("df_caov_pct_all", result)

    def test_robyn_immcarr(self):
        # Setup mock data
        pareto_data = ParetoData(
            pd.DataFrame({'solID': [1], 'coef': [0.5], 'rn': ['test']}),
            pd.DataFrame(),
            pd.DataFrame(),
            []
        )
        result_hyp_param = pd.DataFrame({'solID': [1]})

        # Run the robyn_immcarr function
        result = self.optimizer.robyn_immcarr(
            pareto_data=pareto_data, result_hyp_param=result_hyp_param
        )

        # Assertions to check the return value
        self.assertIsInstance(result, pd.DataFrame)

    def test_extract_hyperparameter(self):
        # Setup mock data
        hyp_param_sam = pd.DataFrame({
            'media_alphas': [0.1, 0.2],
            'media_gammas': [0.3, 0.4],
            'media_thetas': [0.5, 0.6],  # Include this if using GEOMETRIC adstock
            'media_shapes': [0.7, 0.8],  # Include this if using WEIBULL_CDF or WEIBULL_PDF
            'media_scales': [0.9, 1.0],  # Include this if using WEIBULL_CDF or WEIBULL_PDF
        })

        # Run the _extract_hyperparameter function
        result = self.optimizer._extract_hyperparameter(hypParamSam=hyp_param_sam)

        # Assertions to check the return value
        self.assertIsInstance(result, Hyperparameters)

    def test_model_decomp(self):
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
        result = self.optimizer._model_decomp(inputs=inputs)

        # Assertions to check the return value
        self.assertIsInstance(result, dict)
        self.assertIn("xDecompVec", result)
        self.assertIn("mediaDecompImmediate", result)
        self.assertIn("mediaDecompCarryover", result)


# if __name__ == "__main__":
#     unittest.main()
