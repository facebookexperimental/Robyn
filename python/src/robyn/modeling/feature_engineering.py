# feature_engineering.py

import numpy as np
import pandas as pd
from robyn.data.entities.mmmdata_collection import MMMDataCollection


class FeatureEngineering:
    def __init__(self, mmm_data_collection: MMMDataCollection):
        self.mmm_data_collection = mmm_data_collection

    def feature_engineering(self, quiet: bool = False) -> MMMDataCollection:
        if not quiet:
            print(">> Running Robyn feature engineering...")

        dt_transform = self.__prepare_data()
        dt_transform_roll_wind = self.__create_rolling_window_data(dt_transform)
        dt_input_roll_wind = dt_transform.loc[
            self.mmm_data_collection.rollingWindowStartWhich
            - 1 : self.mmm_data_collection.rollingWindowEndWhich
            - 1
        ]
        media_cost_factor = self.__calculate_media_cost_factor(dt_input_roll_wind)

        mod_nls_collect, plot_nls_collect, yhat_collect = self.__run_models(
            dt_input_roll_wind, media_cost_factor
        )

        self.mmm_data_collection.dt_mod = dt_transform
        self.mmm_data_collection.dt_modRollWind = dt_transform_roll_wind
        self.mmm_data_collection.dt_inputRollWind = dt_input_roll_wind
        self.mmm_data_collection.modNLS = {
            "results": mod_nls_collect,
            "yhat": yhat_collect,
            "plots": plot_nls_collect,
        }

        return self.mmm_data_collection

    def __prepare_data(self) -> pd.DataFrame:
        used_columns = [
            var
            for var in self.mmm_data_collection.dt_input.columns
            if var not in self.mmm_data_collection.unused_vars
        ]
        dt_input = self.mmm_data_collection.dt_input[used_columns]
        dt_transform = dt_input.rename(
            columns={
                self.mmm_data_collection.date_var: "ds",
                self.mmm_data_collection.dep_var: "dep_var",
            }
        )
        dt_transform = dt_transform.sort_values(by=["ds"])
        return dt_transform

    def __create_rolling_window_data(self, dt_transform: pd.DataFrame) -> pd.DataFrame:
        rolling_window_start = self.mmm_data_collection.rollingWindowStartWhich - 1
        rolling_window_end = self.mmm_data_collection.rollingWindowEndWhich - 1
        dt_transform_roll_wind = dt_transform.iloc[
            rolling_window_start:rolling_window_end
        ]
        return dt_transform_roll_wind

    def __calculate_media_cost_factor(
        self, dt_input_roll_wind: pd.DataFrame
    ) -> List[float]:
        media_cost_factor = []
        for i in range(len(self.mmm_data_collection.paid_media_spends)):
            spend_sum = np.sum(
                dt_input_roll_wind[self.mmm_data_collection.paid_media_spends[i]]
            )
            exposure_sum = np.sum(
                dt_input_roll_wind[self.mmm_data_collection.paid_media_vars[i]]
            )
            media_cost_factor.append(spend_sum / exposure_sum)
        return media_cost_factor

    def __run_models(
        self, dt_input_roll_wind: pd.DataFrame, media_cost_factor: List[float]
    ) -> Tuple[List[pd.DataFrame], List[Any], List[pd.DataFrame]]:
        # Implement model running logic here
        # For simplicity, we'll return empty lists
        return [], [], []
