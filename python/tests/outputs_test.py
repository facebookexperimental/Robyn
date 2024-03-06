# import pytest
import unittest
import python.outputs
import pandas as pd
import os
import re
import json
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from plotnine import ggplot, aes, geom_point

class OutputsTest(unittest.TestCase):
    def test_robyn_outputs():
        """
        Test case for the robyn_outputs function.

        This test case verifies the output of the robyn_outputs function by performing assertions on the returned result.

        Returns:
            None
        """
        # Create dummy input data
        pandas2ri.activate()
        inputCollect = robjects.r('readRDS("./input_collect.rds")')
        outputModels = robjects.r('readRDS("./output_models.rds")')
        modNLS = inputCollect[40]
        modNLS_plots = modNLS[2]

        # Convert the R object to a Python list
        with localconverter(pandas2ri.converter):
            dt_input = pd.DataFrame(inputCollect[0])
            dt_holidays = pd.DataFrame(inputCollect[1])
            dt_mod = pd.DataFrame(inputCollect[2])
            dt_modRollWind = pd.DataFrame(inputCollect[3])
            dt_inputRollWind = pd.DataFrame(inputCollect[39])
            hyperparameters = dict(inputCollect[35])
            modNLS_results = pd.DataFrame(modNLS[0])
            modNLS_yhat = pd.DataFrame(modNLS[1])
        for gg_R in modNLS_plots:
            data = gg_R[0]
            y = data[1]
            x = data[2]
            p = ggplot() + geom_point(data=data, aes=aes(x=x, y=y))
            p.show()

        dt_input.DATE = pd.to_datetime(dt_input.DATE, unit='d', origin='unix').astype('datetime64[ns]')
        dt_holidays.ds = pd.to_datetime(dt_holidays.ds, unit='d', origin='unix').astype('datetime64[ns]')
        dt_mod.ds = pd.to_datetime(dt_mod.ds, unit='d', origin='unix').astype('datetime64[ns]')
        dt_modRollWind.ds = pd.to_datetime(dt_modRollWind.ds, unit='d', origin='unix').astype('datetime64[ns]')
        dt_inputRollWind.DATE = pd.to_datetime(dt_inputRollWind.DATE, unit='d', origin='unix').astype('datetime64[ns]')


        lst = inputCollect.tolist()
        df = pd.DataFrame(robjects.globalenv[lst['dt_input']])

        # Call the function
        result = robyn_outputs(
            inputCollect,
            outputModels,
            # Automatically pick how many Pareto fronts to fill
            pareto_fronts='auto',
            # Set to 100 top Pareto models for clustering
            min_candidates=100,
            # Calibration constraint
            calibration_constraint=0.1,
            # Export results to CSV files
            csv_out='pareto',
            # Cluster similar models by ROAS
            clusters=True,
            # Create files locally
            export=True,
            # Path for plots exports and files creation
            plot_folder='~/Desktop',
            # Set to FALSE to deactivate plotting and saving model one-pagers
            plot_pareto=False
        )

        # Perform assertions on the output
        assert isinstance(result, dict)
        assert 'resultHypParam' in result
        assert 'xDecompAgg' in result
        assert 'resultCalibration' in result
        assert 'plotDataCollect' in result
        assert 'df_caov_pct' in result

        assert isinstance(result['resultHypParam'], pd.DataFrame)
        assert isinstance(result['xDecompAgg'], pd.DataFrame)
        assert isinstance(result['resultCalibration'], (pd.DataFrame, type(None)))
        assert isinstance(result['plotDataCollect'], (pd.DataFrame, type(None)))
        assert isinstance(result['df_caov_pct'], pd.DataFrame)


    # def test_print_robyn_outputs(capsys):
    #     """
    #     Test function for printing Robyn outputs.

    #     Args:
    #         capsys: pytest fixture for capturing stdout and stderr

    #     Returns:
    #         None
    #     """
    #     # Create dummy input data
    #     x = {
    #         'plot_folder': '/path/to/plot',
    #         'calibration_constraint': True,
    #         'hyper_fixed': False,
    #         'pareto_fronts': 3,
    #         'allSolutions': ['solution1', 'solution2', 'solution3'],
    #         'clusters': {
    #             'n_clusters': 5,
    #             'models': {
    #                 'solID': ['cluster1', 'cluster2', 'cluster3']
    #             }
    #         }
    #     }

    #     # Call the function
    #     print_robyn_outputs(x)

    #     # Capture the printed output
    #     captured = capsys.readouterr()

    #     # Perform assertions on the output
    #     assert "Plot Folder: /path/to/plot" in captured.out
    #     assert "Calibration Constraint: True" in captured.out
    #     assert "Hyper-parameters fixed: False" in captured.out
    #     assert "Pareto-front (3) All solutions (3): solution1, solution2, solution3" in captured.out
    #     assert "Clusters (k = 5): cluster1, cluster2, cluster3" in captured.out

    # def test_robyn_csv():
    #     """
    #     Test function for exporting CSV files using the robyn_csv function.

    #     This function creates dummy input and output data, calls the robyn_csv function,
    #     and performs assertions on the exported CSV files.

    #     Args:
    #         None

    #     Returns:
    #         None
    #     """
    #     # Create dummy input data
    #     input_collect = {
    #         'dt_input': pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]}),
    #     }
    #     output_collect = {
    #         'allPareto': {
    #             'resultHypParam': pd.DataFrame({'param1': [1, 2, 3], 'param2': [4, 5, 6]}),
    #             'xDecompAgg': pd.DataFrame({'agg1': [1, 2, 3], 'agg2': [4, 5, 6]}),
    #             'resultCalibration': pd.DataFrame({'calib1': [1, 2, 3], 'calib2': [4, 5, 6]}),
    #         },
    #         'plot_folder': '/path/to/plot/folder',
    #         'mediaVecCollect': pd.DataFrame({'media1': [1, 2, 3], 'media2': [4, 5, 6]}),
    #         'xDecompVecCollect': pd.DataFrame({'decomp1': [1, 2, 3], 'decomp2': [4, 5, 6]}),
    #     }
    #     csv_out = 'pareto'
    #     export = True
    #     calibrated = False

    #     # Call the function
    #     robyn_csv(input_collect, output_collect, csv_out, export, calibrated)

    #     # Perform assertions on the exported CSV files
    #     assert os.path.exists(os.path.join(output_collect['plot_folder'], 'pareto_hyperparameters.csv'))
    #     assert os.path.exists(os.path.join(output_collect['plot_folder'], 'pareto_aggregated.csv'))
    #     assert not os.path.exists(os.path.join(output_collect['plot_folder'], 'pareto_calibration.csv'))

    #     assert os.path.exists(os.path.join(output_collect['plot_folder'], 'raw_data.csv'))
    #     assert os.path.exists(os.path.join(output_collect['plot_folder'], 'pareto_media_transform_matrix.csv'))
    #     assert os.path.exists(os.path.join(output_collect['plot_folder'], 'pareto_alldecomp_matrix.csv'))
if __name__ == '__main__':
    OutputsTest.test_robyn_outputs()
    unittest.main(argv=[__file__])
# Run the tests
# test_robyn_outputs()
# pytest.main(["-v", "--tb=line", "~/Documents/GitHub/Robyn/python/python/tests/outputs_allocator.py"])
