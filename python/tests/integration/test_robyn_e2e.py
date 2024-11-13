# # pyre-strict
# import pytest
# import pandas as pd
# from unittest.mock import MagicMock, patch
# from pathlib import Path
# from robyn.data.entities.mmmdata import MMMData
# from robyn.data.entities.enums import AdstockType
# from robyn.data.entities.holidays_data import HolidaysData
# from robyn.data.entities.hyperparameters import Hyperparameters, ChannelHyperparameters
# from robyn.modeling.feature_engineering import FeatureEngineering
# from robyn.modeling.model_executor import ModelExecutor
# from robyn.modeling.entities.modelrun_trials_config import TrialsConfig
# from robyn.modeling.entities.enums import NevergradAlgorithm, Models
# from robyn.modeling.pareto.pareto_optimizer import ParetoOptimizer
# from robyn.allocator.budget_allocator import BudgetAllocator
# from robyn.allocator.entities.allocation_config import AllocationConfig
# from robyn.allocator.entities.allocation_constraints import AllocationConstraints
# from robyn.allocator.entities.enums import OptimizationScenario, ConstrMode
# from robyn.visualization.allocator_plotter import AllocationPlotter


# @pytest.fixture(scope="session")
# def test_data():
#     resources_path = Path("python/src/robyn/tutorials/resources")
#     dt_simulated = pd.read_csv(resources_path / "dt_simulated_weekly.csv", parse_dates=["DATE"])
#     dt_holidays = pd.read_csv(resources_path / "dt_prophet_holidays.csv", parse_dates=["ds"])
#     return dt_simulated, dt_holidays


# @pytest.fixture(scope="session")
# def mmm_data(test_data):
#     dt_simulated_weekly, _ = test_data
#     mmm_data_spec = MMMData.MMMDataSpec(
#         dep_var="revenue",
#         dep_var_type="revenue",
#         date_var="DATE",
#         context_vars=["competitor_sales_B", "events"],
#         paid_media_spends=["tv_S", "ooh_S", "print_S", "facebook_S", "search_S"],
#         paid_media_vars=["tv_S", "ooh_S", "print_S", "facebook_I", "search_clicks_P"],
#         organic_vars=["newsletter"],
#         window_start="2016-01-01",
#         window_end="2018-12-31",
#     )
#     return MMMData(data=dt_simulated_weekly, mmmdata_spec=mmm_data_spec)


# @pytest.fixture(scope="session")
# def hyperparameters():
#     return Hyperparameters(
#         {
#             "facebook_S": ChannelHyperparameters(alphas=[0.5, 3], gammas=[0.3, 1], thetas=[0, 0.3]),
#             "print_S": ChannelHyperparameters(alphas=[0.5, 3], gammas=[0.3, 1], thetas=[0.1, 0.4]),
#             "tv_S": ChannelHyperparameters(alphas=[0.5, 3], gammas=[0.3, 1], thetas=[0.3, 0.8]),
#             "search_S": ChannelHyperparameters(alphas=[0.5, 3], gammas=[0.3, 1], thetas=[0, 0.3]),
#             "ooh_S": ChannelHyperparameters(alphas=[0.5, 3], gammas=[0.3, 1], thetas=[0.1, 0.4]),
#             "newsletter": ChannelHyperparameters(alphas=[0.5, 3], gammas=[0.3, 1], thetas=[0.1, 0.4]),
#         },
#         adstock=AdstockType.GEOMETRIC,
#         lambda_=0.0,
#         train_size=[0.5, 0.8],
#     )


# @pytest.fixture(scope="session")
# def holidays_data(test_data):
#     _, dt_holidays = test_data
#     return HolidaysData(
#         dt_holidays=dt_holidays,
#         prophet_vars=["trend", "season", "holiday"],
#         prophet_country="DE",
#         prophet_signs=["default", "default", "default"],
#     )


# @pytest.fixture(scope="session")
# def featurized_mmm_data(mmm_data, hyperparameters, holidays_data):
#     feature_engineering = FeatureEngineering(mmm_data, hyperparameters, holidays_data)
#     return feature_engineering.perform_feature_engineering()


# @pytest.fixture(scope="session")
# def model_outputs(mmm_data, holidays_data, hyperparameters, featurized_mmm_data):
#     """Fixture to create and store model outputs"""
#     model_executor = ModelExecutor(
#         mmmdata=mmm_data,
#         holidays_data=holidays_data,
#         hyperparameters=hyperparameters,
#         calibration_input=None,
#         featurized_mmm_data=featurized_mmm_data,
#     )

#     trials_config = TrialsConfig(iterations=10, trials=5)

#     output_models = model_executor.model_run(
#         trials_config=trials_config,
#         ts_validation=False,
#         add_penalty_factor=False,
#         rssd_zero_penalty=True,
#         cores=8,
#         nevergrad_algo=NevergradAlgorithm.TWO_POINTS_DE,
#         intercept=True,
#         intercept_sign="non_negative",
#         model_name=Models.RIDGE,
#     )
#     return output_models


# @pytest.fixture(scope="session")
# def pareto_result(mmm_data, hyperparameters, featurized_mmm_data, holidays_data, model_outputs):
#     """Fixture to create and store pareto optimization results"""
#     pareto_optimizer = ParetoOptimizer(mmm_data, model_outputs, hyperparameters, featurized_mmm_data, holidays_data)
#     return pareto_optimizer.optimize(pareto_fronts="auto", min_candidates=1)


# @pytest.mark.skip(reason="Disabling this test because it was implemented incorrectly")
# def test_feature_engineering(featurized_mmm_data):
#     assert featurized_mmm_data is not None, "Feature engineering failed, no features generated."


# @pytest.mark.skip(reason="Disabling this test because it was implemented incorrectly")
# def test_model_execution(model_outputs):
#     assert model_outputs is not None, "Model execution failed, no output models returned."
#     assert model_outputs.select_id is not None, "No best models found"


# @pytest.mark.skip(reason="Disabling this test because it was implemented incorrectly")
# def test_pareto_optimization(pareto_result):
#     assert pareto_result is not None, "Pareto optimization failed, no result returned."
#     assert len(pareto_result.pareto_solutions) > 0, "No Pareto solutions found."


# @pytest.mark.skip(reason="Disabling this test because it was implemented incorrectly")
# def test_budget_allocation(mmm_data, featurized_mmm_data, model_outputs, pareto_result):
#     # Budget Allocation
#     select_model = next(iter(pareto_result.pareto_solutions))
#     allocator = BudgetAllocator(
#         mmm_data=mmm_data,
#         featurized_mmm_data=featurized_mmm_data,
#         model_outputs=model_outputs,
#         pareto_result=pareto_result,
#         select_model=select_model,
#     )

#     channel_constraints = AllocationConstraints(
#         channel_constr_low={"tv_S": 0.7, "ooh_S": 0.7, "print_S": 0.7, "facebook_S": 0.7, "search_S": 0.7},
#         channel_constr_up={"tv_S": 1.2, "ooh_S": 1.5, "print_S": 1.5, "facebook_S": 1.5, "search_S": 1.5},
#         channel_constr_multiplier=3.0,
#     )

#     max_response_config = AllocationConfig(
#         scenario=OptimizationScenario.MAX_RESPONSE,
#         constraints=channel_constraints,
#         date_range="last",
#         total_budget=None,
#         maxeval=100000,
#         optim_algo="SLSQP_AUGLAG",
#         constr_mode=ConstrMode.EQUALITY,
#         plots=True,
#     )

#     result = allocator.allocate(max_response_config)

#     assert result is not None, "Budget allocation failed, no result returned."
#     assert len(result.optimal_allocations) > 0, "No optimal allocations found."
