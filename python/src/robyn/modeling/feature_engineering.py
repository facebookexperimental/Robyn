from typing import List, Dict, Union, Tuple, Optional
import pandas as pd
import numpy as np
from plotnine import ggplot, aes, geom_point, geom_line, labs, theme_gray, scale_x_continuous, scale_y_continuous

from robyn.data.entities.mmmdata_collection import MMMDataCollection

class FeatureEngineering:
    def __init__(self, mmm_data_collection: MMMDataCollection) -> None:
        self.mmm_data_collection = mmm_data_collection

    #inputs.robyn_engineering
    def feature_engineering(self, quiet: bool = False) -> MMMDataCollection:
        """Performs feature engineering for the Robyn model.

        Args:
            quiet (bool, optional): If True, suppresses the printing of recommendations and warnings. Defaults to False.

        Returns:
            Dict[str, Union[pd.DataFrame, List[str], int, str, Dict[str, Optional[Union[pd.DataFrame, List[pd.DataFrame]]]]]]: The updated input dictionary with engineered features.
        """
        print(">> Running Robyn feature engineering...")

        # dt_transform = self.__prepare_data()
        # dt_transform_roll_wind = self.__create_rolling_window_data(dt_transform)
        # dt_input_roll_wind = dt_transform.loc[(self.mmm_data_collection['rollingWindowStartWhich']-1):(self.mmm_data_collection['rollingWindowEndWhich']-1)]
        # media_cost_factor = self.__calculate_media_cost_factor(dt_input_roll_wind)

        # mod_nls_collect, plot_nls_collect, yhat_collect = self.__run_models(dt_input_roll_wind, media_cost_factor)

        # if mod_nls_collect:
        #     mod_nls_collect = pd.concat(mod_nls_collect)
        #     yhat_collect = pd.concat(yhat_collect)
        #     repeat_factor = len(yhat_collect) // len(dt_transform_roll_wind)
        #     yhat_collect['ds'] = dt_transform_roll_wind['ds'].repeat(repeat_factor).reset_index(drop=True)
        # else:
        #     mod_nls_collect = None
        #     plot_nls_collect = None
        #     yhat_collect = None

        self.mmm_data_collection['dt_mod'] = dt_transform
        self.mmm_data_collection['dt_modRollWind'] = dt_transform_roll_wind
        self.mmm_data_collection['dt_inputRollWind'] = dt_input_roll_wind
        self.mmm_data_collection['modNLS'] = {
            'results': mod_nls_collect,
            'yhat': yhat_collect,
            'plots': plot_nls_collect
        }

        return self.mmm_data_collection

    def __prepare_data(self) -> pd.DataFrame:
        used_columns = [var for var in self.mmm_data_collection['dt_input'].columns if var not in self.mmm_data_collection['unused_vars']]
        dt_input = self.mmm_data_collection['dt_input'][used_columns]
        dt_transform = dt_input.rename(columns={self.mmm_data_collection['date_var']: 'ds', self.mmm_data_collection['dep_var']: 'dep_var'})
        dt_transform = dt_transform.sort_values(by=['ds'])
        return dt_transform

    def __create_rolling_window_data(self, dt_transform: pd.DataFrame) -> pd.DataFrame:
        rolling_window_start = self.mmm_data_collection['rollingWindowStartWhich'] - 1
        rolling_window_end = self.mmm_data_collection['rollingWindowEndWhich'] - 1
        dt_transform_roll_wind = dt_transform.iloc[rolling_window_start:rolling_window_end]
        return dt_transform_roll_wind

    def __calculate_media_cost_factor(self, dt_input_roll_wind: pd.DataFrame) -> List[float]:
        media_cost_factor = []
        for i in range(len(self.mmm_data_collection['paid_media_spends'])):
            spend_sum = np.sum(dt_input_roll_wind[self.mmm_data_collection['paid_media_spends'][i]])
            exposure_sum = np.sum(dt_input_roll_wind[self.mmm_data_collection['paid_media_vars'][i]])
            media_cost_factor.append(spend_sum / exposure_sum)
        return media_cost_factor

    def __fit_spend_exposure(self, dt_spend_mod_input: pd.DataFrame, media_cost_factor: float, media_var: str) -> Dict[str, Union[pd.DataFrame, Dict[str, Union[float, None]]]]:
        # Placeholder for actual implementation
        return {
            'res': {
                'aic_nls': pd.Series([0.0]),
                'rsq_nls': pd.Series([0.0]),
                'aic_lm': pd.Series([0.0]),
                'rsq_lm': pd.Series([0.0])
            },
            'yhatNLS': pd.Series([0.0]),
            'yhatLM': pd.Series([0.0]),
            'data': {
                'exposure': pd.Series([0.0]),
                'spend': pd.Series([0.0])
            }
        }

    def __run_models(self, dt_input_roll_wind: pd.DataFrame, media_cost_factor: List[float]) -> Tuple[List[pd.DataFrame], List[ggplot], List[pd.DataFrame]]:
        mod_nls_collect = []
        plot_nls_collect = []
        yhat_collect = []

        exposure_selector = [val != self.mmm_data_collection['paid_media_vars'][i] for i, val in enumerate(self.mmm_data_collection['paid_media_spends'])]

        for i in range(self.mmm_data_collection['mediaVarCount']):
            if exposure_selector[i]:
                dt_spend_mod_input = dt_input_roll_wind[[self.mmm_data_collection['paid_media_spends'][i], self.mmm_data_collection['paid_media_vars'][i]]]
                results = self.__fit_spend_exposure(dt_spend_mod_input, media_cost_factor[i], self.mmm_data_collection['paid_media_vars'][i])
                mod = results['res']

                dt_plot_nls = pd.DataFrame({
                    'channel': self.mmm_data_collection['paid_media_vars'][i],
                    'yhat_nls': results['yhatNLS'] if exposure_selector[i] else results['yhatLM'],
                    'yhat_lm': results['yhatLM'],
                    'y': results['data']['exposure'],
                    'x': results['data']['spend'],
                    'caption': f"nls: AIC = {mod['aic_nls'].values[0]} | R2 = {mod['rsq_nls'].values[0]}\nlm: AIC = {mod['aic_lm'].values[0]} | R2 = {mod['rsq_lm'].values[0]}"
                })

                dt_plot_nls = pd.melt(dt_plot_nls, id_vars=['channel', 'y', 'x'], value_vars=['yhat_nls', 'yhat_lm'], var_name='models', value_name='yhat', ignore_index=False)

                models_plot = (
                    ggplot(dt_plot_nls) +
                    aes(x="x", y="y", color="models") +
                    geom_point() +
                    geom_line(aes(y="yhat", x="x", color="models")) +
                    labs(
                        title="Exposure-Spend Models Fit Comparison",
                        x=f"Spend [{self.mmm_data_collection['paid_media_spends'][i]}]",
                        y=f"Exposure [{self.mmm_data_collection['paid_media_vars'][i]}]",
                        caption=dt_plot_nls['caption'].iloc[0]
                    ) +
                    theme_gray() +
                    scale_x_continuous() +
                    scale_y_continuous()
                )

                mod_nls_collect.append(mod)
                plot_nls_collect.append(models_plot)
                yhat_collect.append(dt_plot_nls)

        return mod_nls_collect, plot_nls_collect, yhat_collect
