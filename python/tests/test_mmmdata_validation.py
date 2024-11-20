import unittest
import pandas as pd
import numpy as np

from src.robyn.data.entities.mmmdata import MMMData
from src.robyn.data.validation.mmmdata_validation import MMMDataValidation


class TestMMMDataValidation(unittest.TestCase):

    def setUp(self):
        # Create a sample MMMData object for testing
        data = pd.DataFrame(
            {
                "date": pd.date_range(start="2022-01-01", periods=10),
                "revenue": [100, 120, 110, 130, 140, 150, 160, 170, 180, 190],
                "tv_spend": [50, 60, 55, 65, 70, 75, 80, 85, 90, 95],
                "radio_spend": [30, 35, 32, 38, 40, 42, 45, 48, 50, 52],
                "temperature": [20, 22, 21, 23, 24, 25, 26, 27, 28, 29],
            }
        )

        mmm_data_spec = MMMData.MMMDataSpec(
            dep_var="revenue",
            date_var="date",
            paid_media_spends=["tv_spend", "radio_spend"],
            context_vars=["temperature"],
        )

        self.mmm_data = MMMData(data, mmm_data_spec)
        self.validation = MMMDataValidation(self.mmm_data)

    def test_check_missing_and_infinite(self):
        result = self.validation.check_missing_and_infinite()
        self.assertTrue(result.status)
        self.assertFalse(result.error_details)

        # Introduce missing and infinite values
        self.mmm_data.data.loc[0, "tv_spend"] = np.nan
        self.mmm_data.data.loc[1, "radio_spend"] = np.inf

        result = self.validation.check_missing_and_infinite()
        self.assertFalse(result.status)
        self.assertIn("missing", result.error_details)
        self.assertIn("infinite", result.error_details)

    def test_check_no_variance(self):
        result = self.validation.check_no_variance()
        self.assertTrue(result.status)
        self.assertFalse(result.error_details)

        # Introduce a column with no variance
        self.mmm_data.data["constant"] = 1

        result = self.validation.check_no_variance()
        self.assertFalse(result.status)
        self.assertIn("no_variance", result.error_details)

    def test_check_variable_names(self):
        result = self.validation.check_variable_names()
        self.assertTrue(result.status)
        self.assertFalse(result.error_details)

        # Introduce a duplicate and an invalid variable name
        self.mmm_data.mmmdata_spec.paid_media_spends.append("revenue")
        self.mmm_data.mmmdata_spec.context_vars.append("invalid name")

        result = self.validation.check_variable_names()
        self.assertFalse(result.status)
        self.assertIn("duplicates", result.error_details)
        self.assertIn("invalid", result.error_details)

    def test_check_date_variable(self):
        result = self.validation.check_date_variable()
        self.assertTrue(result.status)
        self.assertFalse(result.error_details)

        # Test with 'auto' date variable
        self.mmm_data.mmmdata_spec.date_var = "auto"
        result = self.validation.check_date_variable()
        self.assertFalse(result.status)
        self.assertIn("date_variable", result.error_details)

        # Test with non-existent date variable
        self.mmm_data.mmmdata_spec.date_var = "non_existent_date"
        result = self.validation.check_date_variable()
        self.assertFalse(result.status)
        self.assertIn("date_variable", result.error_details)

    def test_check_dependent_variables(self):
        result = self.validation.check_dependent_variables()
        self.assertTrue(result.status)
        self.assertFalse(result.error_details)

        # Test with non-existent dependent variable
        self.mmm_data.mmmdata_spec.dep_var = "non_existent_var"
        result = self.validation.check_dependent_variables()
        self.assertFalse(result.status)
        self.assertIn("dependent_variable", result.error_details)

        # Test with non-numeric dependent variable
        self.mmm_data.mmmdata_spec.dep_var = "date"
        result = self.validation.check_dependent_variables()
        self.assertFalse(result.status)
        self.assertIn("dependent_variable", result.error_details)

    def test_validate(self):
        results = self.validation.validate()
        self.assertEqual(len(results), 5)  # Ensure all 5 validations are performed
        self.assertTrue(all(result.status for result in results))
