import unittest
import pandas as pd
import numpy as np
from robyn.modeling.entities.feature_engineering_data import FeatureEngineeringInputData, FeatureEngineeringOutputData
from robyn.modeling.feature_engineering import FeatureEngineering

class TestFeatureEngineering(unittest.TestCase):
    def setUp(self):
        # Create mock input data
        data = {
            'date': pd.date_range(start='2021-01-01', periods=10, freq='D'),
            'spend1': np.random.rand(10) * 100,
            'var1': np.random.rand(10) * 50,
            'spend2': np.random.rand(10) * 200,
            'var2': np.random.rand(10) * 100,
            'dep_var': np.random.rand(10) * 300
        }
        dt_input = pd.DataFrame(data)
        
        self.input_data = FeatureEngineeringInputData(
            dt_input=dt_input,
            date_var='date',
            dep_var='dep_var',
            dep_var_type='continuous',
            paid_media_spends=['spend1', 'spend2'],
            paid_media_vars=['var1', 'var2'],
            paid_media_signs=[1, 1],
            context_vars=[],
            context_signs=[],
            organic_vars=[],
            organic_signs=[],
            factor_vars=[],
            dt_holidays=pd.DataFrame(),
            prophet_vars=[],
            prophet_signs=[],
            prophet_country='US',
            adstock='geometric',
            hyperparameters={},
            window_start='2021-01-01',
            window_end='2021-01-10',
            calibration_input=pd.DataFrame(),
            json_file=None
        )

    def test_feature_engineering(self):
        # Initialize the FeatureEngineering class
        fe = FeatureEngineering(self.input_data)
        
        # Run the feature engineering process
        output_data = fe.feature_engineering(quiet=True)
        
        # Assertions to verify the output
        self.assertIsInstance(output_data, FeatureEngineeringOutputData)
        self.assertIsInstance(output_data.dt_mod, pd.DataFrame)
        self.assertIsInstance(output_data.dt_modRollWind, pd.DataFrame)
        self.assertIsInstance(output_data.modNLS, dict)
        self.assertIn('results', output_data.modNLS)
        self.assertIn('yhat', output_data.modNLS)
        self.assertIn('plots', output_data.modNLS)

        # Check if the results are not empty
        self.assertGreater(len(output_data.modNLS['results']), 0)
        self.assertGreater(len(output_data.modNLS['yhat']), 0)
        self.assertGreater(len(output_data.modNLS['plots']), 0)

if __name__ == '__main__':
    unittest.main()