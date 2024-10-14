# pyre-strict

from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from functools import partial

from robyn.data.entities.hyperparameters import Hyperparameters
from robyn.data.entities.mmmdata import MMMData
from robyn.modeling.entities.modeloutputs import ModelOutputs
from robyn.modeling.pareto.hill_calculator import HillCalculator
from robyn.modeling.pareto.response_curve import ResponseCurveCalculator, ResponseOutput
from robyn.modeling.pareto.immediate_carryover import ImmediateCarryoverCalculator
from robyn.modeling.pareto.pareto_utils import ParetoUtils


@dataclass
class ParetoResult:
    """
    Holds the results of Pareto optimization for marketing mix models.

    Attributes:
        pareto_solutions (List[str]): List of solution IDs that are Pareto-optimal.
        pareto_fronts (int): Number of Pareto fronts considered in the optimization.
        result_hyp_param (pd.DataFrame): Hyperparameters of Pareto-optimal solutions.
        x_decomp_agg (pd.DataFrame): Aggregated decomposition results for Pareto-optimal solutions.
        result_calibration (Optional[pd.DataFrame]): Calibration results, if calibration was performed.
        media_vec_collect (pd.DataFrame): Collected media vectors for all Pareto-optimal solutions.
        x_decomp_vec_collect (pd.DataFrame): Collected decomposition vectors for all Pareto-optimal solutions.
        plot_data_collect (Dict[str, pd.DataFrame]): Data for various plots, keyed by plot type.
        df_caov_pct_all (pd.DataFrame): Carryover percentage data for all channels and Pareto-optimal solutions.
    """

    pareto_solutions: List[str]
    pareto_fronts: int
    result_hyp_param: pd.DataFrame
    x_decomp_agg: pd.DataFrame
    result_calibration: Optional[pd.DataFrame]
    media_vec_collect: pd.DataFrame
    x_decomp_vec_collect: pd.DataFrame
    plot_data_collect: Dict[str, pd.DataFrame]
    df_caov_pct_all: pd.DataFrame


@dataclass
class ParetoData:
    decomp_spend_dist: pd.DataFrame
    result_hyp_param: pd.DataFrame
    x_decomp_agg: pd.DataFrame
    pareto_fronts: List[int]

class ParetoOptimizer:
    """
    Performs Pareto optimization on marketing mix models.

    This class orchestrates the Pareto optimization process, including data aggregation,
    Pareto front calculation, response curve calculation, and plot data preparation.

    Attributes:
        mmm_data (MMMData): Input data for the marketing mix model.
        model_outputs (ModelOutputs): Output data from the model runs.
        response_calculator (ResponseCurveCalculator): Calculator for response curves.
        carryover_calculator (ImmediateCarryoverCalculator): Calculator for immediate and carryover effects.
        pareto_utils (ParetoUtils): Utility functions for Pareto-related calculations.
    """

    def __init__(self, mmm_data: MMMData, model_outputs: ModelOutputs, hyper_parameter: Hyperparameters):
        """
        Initialize the ParetoOptimizer.

        Args:
            mmm_data (MMMData): Input data for the marketing mix model.
            model_outputs (ModelOutputs): Output data from the model runs.
            hyper_parameter (Hyperparameters): Hyperparameters for the model runs.
        """
        self.mmm_data = mmm_data
        self.model_outputs = model_outputs
        self.hyper_parameter = hyper_parameter

    def optimize(
        self,
        pareto_fronts: str = "auto",
        min_candidates: int = 100,
        calibration_constraint: float = 0.1,
        calibrated: bool = False,
    ) -> ParetoResult:
        """
        Perform Pareto optimization on the model results.

        This method orchestrates the entire Pareto optimization process, including data aggregation,
        Pareto front calculation, response curve calculation, and preparation of plot data.

        Args:
            pareto_fronts (str): Number of Pareto fronts to consider or "auto" for automatic selection.
            min_candidates (int): Minimum number of candidates to consider when using "auto" Pareto fronts.
            calibration_constraint (float): Constraint for calibration, used if models are calibrated.
            calibrated (bool): Whether the models have undergone calibration.

        Returns:
            ParetoResult: The results of the Pareto optimization process.
        """
        aggregated_data = self._aggregate_model_data(calibrated)
        aggregated_data['result_hyp_param'] = self._compute_pareto_fronts(aggregated_data, pareto_fronts, min_candidates, calibration_constraint)

        pareto_data = self.prepare_pareto_data(aggregated_data, pareto_fronts, min_candidates)
        response_curves = self._compute_response_curves(pareto_data)
        # TODO IMplement
        plotting_data = self._generate_plot_data(aggregated_data, response_curves)

        return ParetoResult(
            pareto_solutions=None, # TODO: plotting_data["solID"].tolist(),
            pareto_fronts=pareto_fronts,
            result_hyp_param=aggregated_data["result_hyp_param"],
            result_calibration=aggregated_data["result_calibration"],
            x_decomp_agg=pareto_data.x_decomp_agg,
            media_vec_collect=None, # TODO: plotting_data["media_vec_collect"],
            x_decomp_vec_collect=None, # TODO: plotting_data["x_decomp_vec_collect"],
            plot_data_collect=None, # TODO: plotting_data,
            df_caov_pct_all=None # TODO: plotting_data.df_caov_pct_all,
        )
    
    def _aggregate_model_data(self, calibrated: bool) -> Dict[str, pd.DataFrame]:
        """
        Aggregate and prepare data from model outputs for Pareto optimization.

        This method combines hyperparameters, decomposition results, and calibration data (if applicable)
        from all model runs into a format suitable for Pareto optimization.

        Args:
            calibrated (bool): Whether the models have undergone calibration.

        Returns:
            Dict[str, pd.DataFrame]: A dictionary containing aggregated data, including:
                - 'result_hyp_param': Hyperparameters for all model runs
                - 'x_decomp_agg': Aggregated decomposition results
                - 'result_calibration': Calibration results (if calibrated is True)
        """
        hyper_fixed = self.model_outputs.hyper_fixed
        # Extract resultCollect from self.model_outputs
        trials = [model for model in self.model_outputs.trials if hasattr(model, 'resultCollect')]

        # Create lists of resultHypParam and xDecompAgg using list comprehension
        resultHypParam_list = [trial.result_hyp_param for trial in self.model_outputs.trials]
        xDecompAgg_list = [trial.x_decomp_agg for trial in self.model_outputs.trials]

        # Concatenate the lists into DataFrames using pd.concat
        resultHypParam = pd.concat(resultHypParam_list, ignore_index=True)
        xDecompAgg = pd.concat(xDecompAgg_list, ignore_index=True)

        if calibrated:
            resultCalibration = pd.concat([pd.DataFrame(trial.liftCalibration) for trial in trials])
            resultCalibration = resultCalibration.rename(columns={'liftMedia': 'rn'})
        else:
            resultCalibration = None
        if not hyper_fixed:
            df_names = [resultHypParam, xDecompAgg]
            if calibrated:
                df_names.append(resultCalibration)
            for df in df_names:
                df['iterations'] = (df['iterNG'] - 1) * self.model_outputs.cores + df['iterPar']
        elif hyper_fixed and calibrated:
            df_names = [resultCalibration]
            for df in df_names:
                df['iterations'] = (df['iterNG'] - 1) * self.model_outputs.cores + df['iterPar']
        
        # Check if recreated model and bootstrap results are available
        if len(xDecompAgg['solID'].unique()) == 1 and 'boot_mean' not in xDecompAgg.columns:
            # Get bootstrap results from model_outputs object
            bootstrap = getattr(self.model_outputs, 'bootstrap', None)
            if bootstrap is not None:
                # Merge bootstrap results with xDecompAgg using left join
                xDecompAgg = pd.merge(xDecompAgg, bootstrap, left_on='rn', right_on='variable')
        
        return {
            'result_hyp_param': resultHypParam,
            'x_decomp_agg': xDecompAgg,
            'result_calibration': resultCalibration,
        }

    
    def _compute_pareto_fronts(
        self, aggregated_data: Dict[str, pd.DataFrame], pareto_fronts: str, min_candidates: int, calibration_constraint: float
    ) -> pd.DataFrame:
        """
        Calculate Pareto fronts from the aggregated model data.

        This method identifies Pareto-optimal solutions based on NRMSE and DECOMP.RSSD
        optimization criteria and assigns them to Pareto fronts.

        Args:
            resultHypParamPareto (pd.DataFrame): DataFrame containing model results,
                                                including 'nrmse' and 'decomp.rssd' columns.
            pareto_fronts (str): Number of Pareto fronts to compute or "auto".

        Returns:
            pd.DataFrame: A dataframe of Pareto-optimal solutions with their corresponding front numbers.
        """
        resultHypParam = aggregated_data['result_hyp_param']
        xDecompAgg = aggregated_data['x_decomp_agg']
        resultCalibration = aggregated_data['result_calibration']

        if not self.model_outputs.hyper_fixed:
            # Filter and group data to calculate coef0
            xDecompAggCoef0 = (xDecompAgg[xDecompAgg['rn'].isin(self.mmm_data.mmmdata_spec.paid_media_spends)]
                            .groupby('solID')['coef']
                            .apply(lambda x: min(x.dropna()) == 0))
            # calculate quantiles
            mape_lift_quantile10 = resultHypParam['mape'].quantile(calibration_constraint)
            nrmse_quantile90 = resultHypParam['nrmse'].quantile(0.9)
            decomprssd_quantile90 = resultHypParam['decomp.rssd'].quantile(0.9)
            # merge resultHypParam with xDecompAggCoef0
            resultHypParam = pd.merge(resultHypParam, xDecompAggCoef0, on='solID', how='left')
            # create a new column 'mape.qt10'
            resultHypParam['mape.qt10'] = (resultHypParam['mape'] <= mape_lift_quantile10) & \
                                        (resultHypParam['nrmse'] <= nrmse_quantile90) & \
                                        (resultHypParam['decomp.rssd'] <= decomprssd_quantile90)
            # filter resultHypParam
            resultHypParamPareto = resultHypParam[resultHypParam['mape.qt10'] == True]
            # calculate Pareto front
            pareto_fronts_df = ParetoOptimizer._pareto_fronts(resultHypParamPareto,
                                        pareto_fronts=pareto_fronts)
            # merge resultHypParamPareto with pareto_fronts_df
            resultHypParamPareto = pd.merge(resultHypParamPareto, pareto_fronts_df, left_on=['nrmse', 'decomp.rssd'], right_on=['x', 'y'])
            resultHypParamPareto = resultHypParamPareto.rename(columns={'pareto_front': 'robynPareto'})
            resultHypParamPareto = resultHypParamPareto.sort_values(['iterNG', 'iterPar', 'nrmse'])[['solID', 'robynPareto']]
            resultHypParamPareto = resultHypParamPareto.groupby('solID').first().reset_index()
            resultHypParam = pd.merge(resultHypParam, resultHypParamPareto, on='solID', how='left')
        else:
            resultHypParam = resultHypParam.assign(mape_qt10=True, robynPareto=1, coef0=np.nan)

        # Calculate combined weighted error scores
        resultHypParam['error_score'] = ParetoUtils.calculate_errors_scores(df=resultHypParam, ts_validation=self.model_outputs.ts_validation)
        return resultHypParam

        
    @staticmethod
    def _pareto_fronts(resultHypParamPareto: pd.DataFrame, pareto_fronts: str) -> pd.DataFrame:
        """
        Calculate Pareto fronts from the aggregated model data.

        This method identifies Pareto-optimal solutions based on NRMSE and DECOMP.RSSD
        optimization criteria and assigns them to Pareto fronts.

        Args:
            resultHypParamPareto (pd.DataFrame): DataFrame containing model results,
                                                including 'nrmse' and 'decomp.rssd' columns.
            pareto_fronts (Union[str, int]): Number of Pareto fronts to calculate or "auto".
        """
        data = pd.DataFrame({'x': resultHypParamPareto['nrmse'], 'y': resultHypParamPareto['decomp.rssd']})
        data = data.sort_values(by=['x', 'y'], ascending=[True, True])
        pareto_fronts_df = pd.DataFrame(columns=['x', 'y', 'pareto_front'])
        i = 1

        while not data.empty and (pareto_fronts == "auto" or i <= int(pareto_fronts)):
            # Identify Pareto front
            is_pareto = data.apply(lambda row: all((row['y'] <= data['y']) & (row['x'] <= data['x'])), axis=1)
            pareto_front = data[is_pareto]
            pareto_front['pareto_front'] = i

            # Append to result dataframe
            pareto_fronts_df = pd.concat([pareto_fronts_df, pareto_front])

            # Remove identified Pareto front from data
            data = data[~is_pareto]
            i += 1
        return pareto_fronts_df.reset_index(drop=True)


    def prepare_pareto_data(self, aggregated_data: Dict[str, pd.DataFrame], 
                            pareto_fronts: str, min_candidates: int) -> ParetoData:
        result_hyp_param = aggregated_data['result_hyp_param']
        x_decomp_agg = aggregated_data['x_decomp_agg']

        # 1. Binding Pareto results
        x_decomp_agg = pd.merge(x_decomp_agg, 
                                result_hyp_param[['robynPareto', 'solID']], 
                                on='solID', how='left')
        
        # Step 1: Collect decomp_spend_dist from each trial and add the trial number
        decomp_spend_dist = pd.concat([
            trial.decomp_spend_dist.assign(trial=trial.trial)
            for trial in self.model_outputs.trials
            if trial.decomp_spend_dist is not None
        ], ignore_index=True)

        # Step 2: Add solID if hyper_fixed is False
        if not self.model_outputs.hyper_fixed:
            decomp_spend_dist['solID'] = (
                decomp_spend_dist['trial'].astype(str) + '_' +
                decomp_spend_dist['iterNG'].astype(str) + '_' +
                decomp_spend_dist['iterPar'].astype(str)
            )

        # Step 4: Left join with resultHypParam
        decomp_spend_dist = pd.merge(
            decomp_spend_dist,
            result_hyp_param[['robynPareto', 'solID']],
            on='solID',
            how='left'
        )

        # 2. Preparing for parallel processing
        # Note: Python parallel processing would be implemented differently
        # You might want to use multiprocessing or concurrent.futures here

        # 3. Determining the number of Pareto fronts
        if self.model_outputs.hyper_fixed or len(result_hyp_param) == 1:
            pareto_fronts = 1

        # 4. Handling automatic Pareto front selection
        if pareto_fronts == "auto":
            n_pareto = result_hyp_param['robynPareto'].notna().sum()
            
            # Check if any trial has lift_calibration
            is_calibrated = any(trial.lift_calibration is not None for trial in self.model_outputs.trials)

            if n_pareto <= min_candidates and len(result_hyp_param) > 1 and not is_calibrated:
                raise ValueError(
                    f"Less than {min_candidates} candidates in pareto fronts. "
                    "Increase iterations to get more model candidates or decrease min_candidates."
                )

            auto_pareto = (result_hyp_param[result_hyp_param['robynPareto'].notna()]
                        .groupby('robynPareto')
                        .agg(n=('solID', 'nunique'))
                        .reset_index()
                        .sort_values('robynPareto'))
            
            auto_pareto['n_cum'] = auto_pareto['n'].cumsum()
            auto_pareto = auto_pareto[auto_pareto['n_cum'] >= min_candidates].iloc[0]

            print(f">> Automatically selected {auto_pareto['robynPareto']} Pareto-fronts "
                f"to contain at least {min_candidates} pareto-optimal models ({auto_pareto['n_cum']})")

            pareto_fronts = int(auto_pareto['robynPareto'])

        # 5. Creating Pareto front vector
        pareto_fronts_vec = list(range(1, pareto_fronts + 1))

        # 6. Filtering data for selected Pareto fronts
        decomp_spend_dist_pareto = decomp_spend_dist[decomp_spend_dist['robynPareto'].isin(pareto_fronts_vec)]
        result_hyp_param_pareto = result_hyp_param[result_hyp_param['robynPareto'].isin(pareto_fronts_vec)]
        x_decomp_agg_pareto = x_decomp_agg[x_decomp_agg['robynPareto'].isin(pareto_fronts_vec)]

        return ParetoData(decomp_spend_dist=decomp_spend_dist_pareto, 
                          result_hyp_param=result_hyp_param_pareto, 
                          x_decomp_agg=x_decomp_agg_pareto, 
                          pareto_fronts=pareto_fronts_vec)

    def run_dt_resp(self, row: pd.Series, paretoData: ParetoData) -> pd.Series:
        """
        Calculate response curves for a given row of Pareto data. 
        This method is used for parallel processing.

        Args:
            row (pd.Series): A row of Pareto data.
            paretoData (ParetoData): Pareto data.

        Returns:
            pd.Series: A row of response curves.
        """
        get_solID = row['solID']
        get_spendname = row['rn']
        startRW = self.mmm_data.mmmdata_spec.rolling_window_start_which
        endRW = self.mmm_data.mmmdata_spec.rolling_window_end_which

        response_calculator = ResponseCurveCalculator(
            mmm_data=self.mmm_data,
            model_outputs=self.model_outputs,
            hyperparameter = self.hyper_parameter)

        response_output: ResponseOutput = response_calculator.calculate_response(
            select_model=get_solID,
            metric_name=get_spendname,
            date_range="all",
            dt_hyppar=paretoData.result_hyp_param,
            dt_coef=paretoData.x_decomp_agg,
            quiet=True
        )

        mean_spend_adstocked = np.mean(response_output.input_total[startRW:endRW])
        mean_carryover = np.mean(response_output.input_carryover[startRW:endRW])
        
        dt_hyppar = paretoData.result_hyp_param[paretoData.result_hyp_param['solID'] == get_solID]
        chn_adstocked = pd.DataFrame({get_spendname: response_output.input_total[startRW:endRW]})
        dt_coef = paretoData.x_decomp_agg[
            (paretoData.x_decomp_agg['solID'] == get_solID) & 
            (paretoData.x_decomp_agg['rn'] == get_spendname)
        ][['rn', 'coef']]

        hill_calculator = HillCalculator(
            mmmdata=self.mmm_data,
            model_outputs=self.model_outputs,
            dt_hyppar=dt_hyppar,
            dt_coef=dt_coef,
            media_spend_sorted=[get_spendname],
            select_model=get_solID,
            chn_adstocked=chn_adstocked
        )
        hills = hill_calculator.get_hill_params()

        mean_response = ParetoUtils.calculate_fx_objective(
            x=row['mean_spend'],
            coeff=hills['coefs_sorted'][0],
            alpha=hills['alphas'][0],
            inflexion=hills['inflexions'][0],
            x_hist_carryover=mean_carryover,
            get_sum=False
        )

        return pd.Series({
            'mean_response': mean_response,
            'mean_spend_adstocked': mean_spend_adstocked,
            'mean_carryover': mean_carryover,
            'rn': row['rn'],
            'solID': row['solID']
        })
    
    def _compute_response_curves(self, pareto_data: ParetoData) -> ParetoData:
        """
        Calculate response curves for Pareto-optimal solutions.

        This method computes response curves for each media channel in each Pareto-optimal solution,
        providing insights into the relationship between media spend and response.

        Args:
            pareto_data (ParetoData): Pareto data.

        Returns:
            ParetoData: Pareto data with updated decomp_spend_dist and x_decomp_agg.
        """
        print(f">>> Calculating response curves for all models' media variables ({len(pareto_data.decomp_spend_dist)})...")

        # Parallel processing
        run_dt_resp_partial = partial(self.run_dt_resp, 
                                      paretoData=pareto_data)

        if self.model_outputs.cores > 1:
            with ProcessPoolExecutor(max_workers=self.model_outputs.cores) as executor:
                futures = [executor.submit(run_dt_resp_partial, row) for _, row in pareto_data.decomp_spend_dist.iterrows()]
                resp_collect = pd.DataFrame([f.result() for f in as_completed(futures)])
        else:
            resp_collect = pareto_data.decomp_spend_dist.apply(run_dt_resp_partial, axis=1)

        # Merge results
        pareto_data.decomp_spend_dist = pd.merge(
            pareto_data.decomp_spend_dist,
            resp_collect,
            on=['solID', 'rn'],
            how='left'
        )
        
        # Calculate ROI and CPA metrics after merging
        pareto_data.decomp_spend_dist['roi_mean'] = pareto_data.decomp_spend_dist['mean_response'] / pareto_data.decomp_spend_dist['mean_spend']
        pareto_data.decomp_spend_dist['roi_total'] = pareto_data.decomp_spend_dist['xDecompAgg'] / pareto_data.decomp_spend_dist['total_spend']
        pareto_data.decomp_spend_dist['cpa_mean'] = pareto_data.decomp_spend_dist['mean_spend'] / pareto_data.decomp_spend_dist['mean_response']
        pareto_data.decomp_spend_dist['cpa_total'] = pareto_data.decomp_spend_dist['total_spend'] / pareto_data.decomp_spend_dist['xDecompAgg']

        pareto_data.x_decomp_agg = pd.merge(
            pareto_data.x_decomp_agg,
            pareto_data.decomp_spend_dist[['rn', 'solID', 'total_spend', 'mean_spend', 'mean_spend_adstocked', 'mean_carryover',
                               'mean_response', 'spend_share', 'effect_share', 'roi_mean', 'roi_total', 'cpa_total']],
            on=['solID', 'rn'],
            how='left'
        )

        return pareto_data
    
    def _generate_plot_data(
        self, aggregated_data: Dict[str, pd.DataFrame], response_curves: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """
        Prepare data for various plots used in the Pareto analysis.

        This method generates data for different types of plots used to visualize and analyze
        the Pareto-optimal solutions, including spend vs. effect comparisons, waterfalls, and more.

        Args:
            pareto_fronts_df (pd.DataFrame): Dataframe of Pareto-optimal solutions.
            response_curves (Dict[str, pd.DataFrame]): Response curves data from _compute_response_curves.

        Returns:
            Dict[str, pd.DataFrame]: A dictionary of dataframes, each containing data for a specific plot type.
        """
        return None
