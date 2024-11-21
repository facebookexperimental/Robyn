# pyre-strict

import unittest
from unittest.mock import Mock, MagicMock
import pandas as pd
from robyn.data.entities.mmmdata import MMMData
from robyn.modeling.entities.modeloutputs import ModelOutputs
from robyn.modeling.pareto.hill_calculator import HillCalculator


class TestHillCalculator(unittest.TestCase):

    def test_empty_media_vec_collect(self):
        mock_media_vec_collect = pd.DataFrame(columns=["type", "solID"])
        mock_mmmdata_spec = type(
            "MockMMMDataSpec", (object,), {"window_start": 1, "window_end": 2}
        )()
        mock_mmmdata = type(
            "MockMMMData", (object,), {"mmmdata_spec": mock_mmmdata_spec}
        )()

        mock_model_outputs = type(
            "MockModelOutputs", (object,), {"media_vec_collect": mock_media_vec_collect}
        )()

        media_spend_sorted = []
        select_model = "model_1"

        # Create an instance of HillCalculator with the mocked data
        hill_calculator = HillCalculator(
            mmmdata=mock_mmmdata,
            model_outputs=mock_model_outputs,
            dt_hyppar=pd.DataFrame(),
            dt_coef=pd.DataFrame(),
            media_spend_sorted=media_spend_sorted,
            select_model=select_model,
        )

        # Invoke _get_chn_adstocked_from_output_collect
        chn_adstocked = hill_calculator._get_chn_adstocked_from_output_collect()

        # Assert that the result chn_adstocked is an empty DataFrame
        self.assertTrue(chn_adstocked.empty)

    def test_no_matching_solID(self):
        # Mock the media_vec_collect DataFrame with non-matching solIDs
        mock_media_vec_collect = pd.DataFrame(
            {
                "type": ["adstockedMedia", "adstockedMedia"],
                "solID": ["non_matching_model_1", "non_matching_model_2"],
                "media_1": [0.1, 0.2],
                "media_2": [0.3, 0.4],
            }
        )

        # Mock the MMMData and ModelOutputs
        mock_mmmdata_spec = type(
            "MockMMMDataSpec", (object,), {"window_start": 1, "window_end": 2}
        )()
        mock_mmmdata = type(
            "MockMMMData", (object,), {"mmmdata_spec": mock_mmmdata_spec}
        )()

        mock_model_outputs = type(
            "MockModelOutputs", (object,), {"media_vec_collect": mock_media_vec_collect}
        )()

        # Create HillCalculator instance with mocked data
        hill_calculator = HillCalculator(
            mmmdata=mock_mmmdata,
            model_outputs=mock_model_outputs,
            dt_hyppar=pd.DataFrame(),
            dt_coef=pd.DataFrame(),
            media_spend_sorted=["media_1", "media_2"],
            select_model="model_1",
        )

        # Invoke the method and assert the result
        chn_adstocked = hill_calculator._get_chn_adstocked_from_output_collect()
        self.assertTrue(chn_adstocked.empty)

    def test_valid_matching_solID_and_window_slicing(self):
        # Mock the ModelOutputs media_vec_collect DataFrame
        mock_media_vec_collect = pd.DataFrame(
            {
                "type": ["adstockedMedia", "adstockedMedia", "adstockedMedia"],
                "solID": ["model_1", "model_1", "model_1"],
                "media_1": [100, 200, 300],
                "media_2": [400, 500, 600],
            }
        )

        # Mock MMMData with window specification
        mock_mmmdata_spec = type(
            "MockMMMDataSpec", (object,), {"window_start": 1, "window_end": 2}
        )()
        mock_mmmdata = type(
            "MockMMMData", (object,), {"mmmdata_spec": mock_mmmdata_spec}
        )()

        # Mock ModelOutputs
        mock_model_outputs = type(
            "MockModelOutputs", (object,), {"media_vec_collect": mock_media_vec_collect}
        )()

        # Initialize the HillCalculator with mock data
        calculator = HillCalculator(
            mmmdata=mock_mmmdata,
            model_outputs=mock_model_outputs,
            dt_hyppar=None,  # Not used in this test
            dt_coef=None,  # Not used in this test
            media_spend_sorted=["media_1", "media_2"],
            select_model="model_1",
        )

        # Invoke the method under test
        chn_adstocked = calculator._get_chn_adstocked_from_output_collect()

        # Expected sliced DataFrame
        expected_chn_adstocked = pd.DataFrame(
            {"media_1": [200, 300], "media_2": [500, 600]}
        ).reset_index(drop=True)

        # Assert the result matches the expected DataFrame
        pd.testing.assert_frame_equal(
            chn_adstocked.reset_index(drop=True), expected_chn_adstocked
        )

    def test_get_hill_params_with_normal_input_data(self):
        # Mock setup
        mock_dt_hyppar = Mock(spec=pd.DataFrame)
        mock_dt_hyppar.filter.return_value = pd.DataFrame(
            {
                "media1_alphas": [0.5],
                "media2_alphas": [0.7],
                "media1_gammas": [0.3],
                "media2_gammas": [0.4],
            }
        )

        adstocked_data = {
            "media1": pd.Series({"min": 100, "max": 200}),
            "media2": pd.Series({"min": 150, "max": 250}),
        }
        mock_chn_adstocked = MagicMock(spec=pd.DataFrame)
        mock_chn_adstocked.__getitem__.side_effect = adstocked_data.get
        mock_dt_coef = pd.DataFrame(
            {"rn": ["media1", "media2"], "coef": ["coef1", "coef2"]}
        )
        # Input preparation
        media_spend_sorted = ["media1", "media2"]
        mock_mmmdata = Mock(spec=MMMData)
        mock_model_outputs = Mock(spec=ModelOutputs)
        # Instantiate HillCalculator with mocks
        hill_calculator = HillCalculator(
            mmmdata=mock_mmmdata,
            model_outputs=mock_model_outputs,
            dt_hyppar=mock_dt_hyppar,
            dt_coef=mock_dt_coef,
            media_spend_sorted=media_spend_sorted,
            select_model="model_1",
            chn_adstocked=mock_chn_adstocked,
        )
        # Invoke function
        result = hill_calculator.get_hill_params()
        # Assertions
        assert result["alphas"] == [0.5, 0.7]
        assert result["inflexions"] == [130.0, 190.0]
        assert result["coefs_sorted"] == ["coef1", "coef2"]

    def test_get_hill_params_with_chn_adstocked_none(self):
        # Create mock data
        mock_chn_adstocked = pd.DataFrame({"media1": [100, 200], "media2": [150, 250]})

        mock_dt_hyppar = pd.DataFrame(
            {
                "media1_alphas": [0.5],
                "media2_alphas": [0.7],
                "media1_gammas": [0.2],
                "media2_gammas": [0.3],
            }
        )

        mock_dt_coef = pd.DataFrame(
            {"rn": ["media1", "media2"], "coef": ["coef1", "coef2"]}
        )

        # Mock MMMData and ModelOutputs
        mock_mmmdata = Mock()
        mock_mmmdata.mmmdata_spec.window_start = 0
        mock_mmmdata.mmmdata_spec.window_end = 1

        mock_model_outputs = Mock()
        mock_model_outputs.media_vec_collect = pd.DataFrame(
            {
                "type": ["adstockedMedia", "adstockedMedia"],
                "solID": ["model1", "model1"],
                "media1": [100, 200],
                "media2": [150, 250],
            }
        )

        # Initialize the HillCalculator with the mock data
        calculator = HillCalculator(
            mmmdata=mock_mmmdata,
            model_outputs=mock_model_outputs,
            dt_hyppar=mock_dt_hyppar,
            dt_coef=mock_dt_coef,
            media_spend_sorted=["media1", "media2"],
            select_model="model1",
            chn_adstocked=None,
        )

        # Mock the method _get_chn_adstocked_from_output_collect
        calculator._get_chn_adstocked_from_output_collect = Mock(
            return_value=mock_chn_adstocked
        )

        # Execute the get_hill_params function
        result = calculator.get_hill_params()

        # Assertions
        assert result["alphas"] == [0.5, 0.7]
        assert result["inflexions"] == [120.0, 180.0]
        assert result["coefs_sorted"] == ["coef1", "coef2"]

    def test_get_hill_params_with_empty_media_spend_sorted(self):
        mock_mmmdata = Mock()
        mock_model_outputs = Mock()
        dt_hyppar = pd.DataFrame(
            {
                "media1_alphas": [0.1],
                "media1_gammas": [0.2],
            }
        )
        dt_coef = pd.DataFrame(
            {
                "rn": ["media1"],
                "coef": [1.0],
            }
        )
        chn_adstocked = pd.DataFrame()

        hill_calculator = HillCalculator(
            mmmdata=mock_mmmdata,
            model_outputs=mock_model_outputs,
            dt_hyppar=dt_hyppar,
            dt_coef=dt_coef,
            media_spend_sorted=[],
            select_model="model_1",
            chn_adstocked=chn_adstocked,
        )

        result = hill_calculator.get_hill_params()

        # Assertions
        self.assertEqual(result["alphas"], [])
        self.assertEqual(result["inflexions"], [])
        self.assertEqual(result["coefs_sorted"], [])
