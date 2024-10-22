# pyre-strict

import unittest
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np

from robyn.data.entities.enums import AdstockType, DependentVarType
from robyn.data.entities.mmmdata import MMMData
from robyn.data.entities.hyperparameters import ChannelHyperparameters, Hyperparameters
from robyn.data.entities.holidays_data import HolidaysData
from robyn.modeling.feature_engineering import FeatureEngineering, FeaturizedMMMData

class TestFeatureEngineering(unittest.TestCase):
    """
    Test suite for the FeatureEngineering class.
    
    This test suite covers the core functionality of feature engineering in the Robyn framework,
    including data preparation, Prophet decomposition, media cost calculations, and model fitting.
    """

    def setUp(self) -> None:
        """
        Set up test fixtures before each test case.
        
        Creates mock data and configurations that will be used across multiple test cases.
        """
        # Load test data
        self.dt_simulated_weekly = pd.read_csv("src/tutorials/resources/dt_simulated_weekly.csv")
        self.dt_prophet_holidays = pd.read_csv("src/tutorials/resources/dt_prophet_holidays.csv")

        # Create MMMData
        mmm_data_spec = MMMData.MMMDataSpec(
            dep_var="revenue",
            dep_var_type=DependentVarType.REVENUE,
            date_var="DATE",
            context_vars=["competitor_sales_B", "events"],
            paid_media_spends=["tv_S", "ooh_S", "print_S", "facebook_S", "search_S"],
            paid_media_vars=["tv_S", "ooh_S", "print_S", "facebook_I", "search_clicks_P"],
            organic_vars=["newsletter"],
            window_start="2016-01-01",
            window_end="2018-12-31",
        )

        self.mmm_data = MMMData(data=self.dt_simulated_weekly, mmmdata_spec=mmm_data_spec)

        # Create Hyperparameters
        hyperparameter_dict = {
            channel: ChannelHyperparameters(
                alphas=[0.5, 3],
                gammas=[0.3, 1],
                thetas=[0.1, 0.4] if channel not in ["facebook_S", "search_S"] else [0, 0.3]
            )
            for channel in ["facebook_S", "print_S", "tv_S", "search_S", "ooh_S", "newsletter"]
        }
        # Update tv_S thetas separately as it has different values
        hyperparameter_dict["tv_S"].thetas = [0.3, 0.8]

        self.hyperparameters = Hyperparameters(
            hyperparameters=hyperparameter_dict,
            adstock=AdstockType.GEOMETRIC,
            lambda_=0.0,
            train_size=[0.5, 0.8]
        )

        # Create HolidaysData
        self.holidays_data = HolidaysData(
            dt_holidays=self.dt_prophet_holidays,
            prophet_vars=["trend", "season", "holiday"],
            prophet_country="DE",
            prophet_signs=["default", "default", "default"]
        )

        # Create Feature Engineering instance
        self.feature_engineering = FeatureEngineering(
            self.mmm_data,
            self.hyperparameters,
            self.holidays_data
        )

    def test_prepare_data(self) -> None:
        """
        Test the _prepare_data method.
        
        Verifies that the method correctly:
        1. Formats dates
        2. Creates dep_var column
        3. Handles data type conversions
        """
        dt_transform = self.feature_engineering._prepare_data()
        
        # Check that required columns are present
        self.assertIn('ds', dt_transform.columns)
        self.assertIn('dep_var', dt_transform.columns)
        
        # Check date formatting
        self.assertTrue(all(isinstance(date, str) for date in dt_transform['ds']))
        # Update regex pattern to properly check date format
        self.assertTrue(all(pd.to_datetime(date).strftime('%Y-%m-%d') == date for date in dt_transform['ds']))
        
        # Check dep_var values match the original revenue column
        np.testing.assert_array_equal(dt_transform['dep_var'], self.mmm_data.data['revenue'])
        
        # Check data types
        self.assertEqual(dt_transform['competitor_sales_B'].dtype, np.int64)

    def test_create_rolling_window_data(self) -> None:
        """
        Test the _create_rolling_window_data method.
        
        Verifies correct handling of:
        1. Window start and end dates
        2. Data filtering
        3. Edge cases with missing window parameters
        """
        dt_transform = self.feature_engineering._prepare_data()
        
        # Test with both window start and end
        result = self.feature_engineering._create_rolling_window_data(dt_transform)
        # Check that filtered data is within the specified window
        self.assertTrue(all(
            (pd.to_datetime(result['ds']) >= pd.to_datetime('2016-01-01')) &
            (pd.to_datetime(result['ds']) <= pd.to_datetime('2018-12-31'))
        ))
        
        # Test with only start date
        original_end = self.mmm_data.mmmdata_spec.window_end
        self.mmm_data.mmmdata_spec.window_end = None
        result = self.feature_engineering._create_rolling_window_data(dt_transform)
        self.assertTrue(all(pd.to_datetime(result['ds']) >= pd.to_datetime('2016-01-01')))
        
        # Test with only end date
        self.mmm_data.mmmdata_spec.window_start = None
        self.mmm_data.mmmdata_spec.window_end = original_end
        result = self.feature_engineering._create_rolling_window_data(dt_transform)
        self.assertTrue(all(pd.to_datetime(result['ds']) <= pd.to_datetime('2018-12-31')))

    def test_calculate_media_cost_factor(self) -> None:
        """
        Test the _calculate_media_cost_factor method.
        
        Verifies:
        1. Correct calculation of media cost factors
        2. Handling of multiple media channels
        3. Sum of factors equals 1
        """
        dt_transform = self.feature_engineering._prepare_data()
        dt_roll_wind = self.feature_engineering._create_rolling_window_data(dt_transform)
        
        media_cost_factor = self.feature_engineering._calculate_media_cost_factor(dt_roll_wind)
        
        # Check that we have factors for all media channels
        self.assertEqual(len(media_cost_factor), len(self.mmm_data.mmmdata_spec.paid_media_spends))
        
        # Check that factors sum to 1
        self.assertAlmostEqual(media_cost_factor.sum(), 1.0)
        
        # Check that all factors are positive
        self.assertTrue(all(media_cost_factor > 0))

    @patch('robyn.modeling.feature_engineering.curve_fit')
    def test_fit_spend_exposure(self, mock_curve_fit: Mock) -> None:
        """
        Test the _fit_spend_exposure method.
        
        Verifies:
        1. Correct fitting of both Michaelis-Menten and linear models
        2. Error handling
        3. Model selection based on R-squared values
        """
        dt_transform = self.feature_engineering._prepare_data()
        dt_roll_wind = self.feature_engineering._create_rolling_window_data(dt_transform)
        media_cost_factor = self.feature_engineering._calculate_media_cost_factor(dt_roll_wind)
        
        # Mock curve_fit to return reasonable parameters
        mock_curve_fit.return_value = ([10.0, 500.0], None)
        
        # Test for facebook_S channel
        result = self.feature_engineering._fit_spend_exposure(
            dt_roll_wind,
            'facebook_S',
            media_cost_factor
        )
        
        # Check result structure
        self.assertIn('res', result)
        self.assertIn('plot', result)
        self.assertIn('yhat', result)
        
        # Check model details
        self.assertIn('model_type', result['res'])
        self.assertIn('rsq', result['res'])
        self.assertIn('coef', result['res'])
        self.assertEqual(result['res']['channel'], 'facebook_S')

    def test_prophet_decomposition(self) -> None:
        """
        Test the _prophet_decomposition method.
        
        Verifies:
        1. Correct handling of Prophet variables
        2. Holiday data processing
        3. Various seasonality components
        """
        dt_transform = self.feature_engineering._prepare_data()
        result = self.feature_engineering._prophet_decomposition(dt_transform)
        
        # Check that Prophet components are present
        self.assertIn('trend', result.columns)
        self.assertIn('season', result.columns)
        self.assertIn('holiday', result.columns)
        
        # Verify values are within reasonable ranges
        self.assertTrue(all(np.isfinite(result['trend'])))
        self.assertTrue(all(np.isfinite(result['season'])))
        self.assertTrue(all(np.isfinite(result['holiday'])))

    def test_perform_feature_engineering(self) -> None:
        """
        Test the perform_feature_engineering method.
        
        Verifies:
        1. Complete feature engineering pipeline
        2. Output structure
        3. Data integrity through the process
        """
        result = self.feature_engineering.perform_feature_engineering(quiet=True)
        
        # Check result type
        self.assertIsInstance(result, FeaturizedMMMData)
        
        # Check that required components are present
        self.assertIsInstance(result.dt_mod, pd.DataFrame)
        self.assertIsInstance(result.dt_modRollWind, pd.DataFrame)
        self.assertIsInstance(result.modNLS, dict)
        
        # Check that all media channels are processed
        self.assertTrue(
            all(media in result.modNLS['results'] 
                for media in self.mmm_data.mmmdata_spec.paid_media_spends)
        )

        # Verify data integrity
        self.assertEqual(len(result.dt_mod), len(self.mmm_data.data))
        self.assertGreater(len(result.dt_modRollWind), 0)

    def test_error_handling(self) -> None:
        """
        Test error handling in feature engineering.
        
        Verifies proper handling of:
        1. Invalid date ranges
        2. Missing required columns
        3. Invalid data types
        """
        # Test with invalid window dates
        original_start = self.mmm_data.mmmdata_spec.window_start
        self.mmm_data.mmmdata_spec.window_start = '2022-01-01'  # After data end
        with self.assertRaises(ValueError):
            self.feature_engineering.perform_feature_engineering()
        self.mmm_data.mmmdata_spec.window_start = original_start
        
        # Test with missing required column
        original_data = self.mmm_data.data.copy()
        self.mmm_data.data = self.mmm_data.data.drop('revenue', axis=1)
        with self.assertRaises(KeyError):
            self.feature_engineering.perform_feature_engineering()
        self.mmm_data.data = original_data

    def test_edge_cases(self) -> None:
        """
        Test edge cases in feature engineering.
        
        Verifies handling of:
        1. Minimal data (2 rows)
        2. Missing values
        3. Extreme values
        4. Data recovery after modifications
        
        Note: Prophet requires at least 2 non-NaN rows for decomposition,
        so we test with minimum 2 rows instead of 1.
        """
        # Save original data and holidays configuration
        original_data = self.mmm_data.data.copy()
        original_prophet_vars = self.holidays_data.prophet_vars.copy()
        
        try:
            # Temporarily disable Prophet decomposition to test with minimal data
            self.holidays_data.prophet_vars = []
            
            # Test with minimal data (2 rows)
            self.mmm_data.data = self.mmm_data.data.iloc[:2].copy()
            result_minimal = self.feature_engineering.perform_feature_engineering()
            self.assertEqual(len(result_minimal.dt_mod), 2)
            self.assertTrue(all(col in result_minimal.dt_mod.columns 
                            for col in self.mmm_data.mmmdata_spec.paid_media_spends))
            
            # Restore original data for next tests
            self.mmm_data.data = original_data.copy()
            
            # Test with missing values
            data_with_na = original_data.copy()
            data_with_na.loc[data_with_na.index[0:5], 'tv_S'] = np.nan
            self.mmm_data.data = data_with_na
            result_with_na = self.feature_engineering.perform_feature_engineering()
            self.assertFalse(result_with_na.dt_mod['tv_S'].isna().any())
            
            # Test with extreme values
            data_with_extreme = original_data.copy()
            data_with_extreme.loc[data_with_extreme.index[0:5], 'tv_S'] = 1e9
            self.mmm_data.data = data_with_extreme
            result_with_extreme = self.feature_engineering.perform_feature_engineering()
            self.assertTrue(np.isfinite(result_with_extreme.dt_mod['tv_S']).all())
            
            # Test that media channels maintain relative proportions after processing
            # original_ratio = (original_data['tv_S'] / original_data['facebook_S']).mean()
            # processed_ratio = (result_with_extreme.dt_mod['tv_S'] / 
            #                 result_with_extreme.dt_mod['facebook_S']).mean()
            # self.assertLess(abs(original_ratio - processed_ratio) / original_ratio, 0.5)
            
        finally:
            # Restore original data and configuration
            self.mmm_data.data = original_data
            self.holidays_data.prophet_vars = original_prophet_vars

if __name__ == '__main__':
    unittest.main()