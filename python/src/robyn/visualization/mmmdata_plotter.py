# pyre-strict

from typing import Dict, Any, List, Optional
import pandas as pd
import matplotlib.pyplot as plt

from robyn.data.entities.mmmdata_collection import MMMDataCollection

class MMMDataPlotter:
    def __init__(self, mmm_data_collection: MMMDataCollection) -> None:
        """
        Initialize the RobynPlotter with an MMMDataCollection.

        :param mmm_data_collection: Collection of MMM data.
        """
        self.mmm_data_collection: MMMDataCollection = mmm_data_collection

    def plot_model_fit(
        self,
        mmm_data_collection: MMMDataCollection,
    ) -> None:
        """
        Plot the model fit, including actual vs predicted values and decomposition of effects.

        :param mmm_data_collection: MMM Data Collection
        """
        pass

    def plot_response_curves(
        self,
        dt_mod: pd.DataFrame,
        x_decomp_vec: pd.DataFrame,
        x_decomp_vec_immediate: pd.DataFrame,
        x_decomp_vec_carryover: pd.DataFrame,
        all_media: List[str],
        paid_media_spends: List[str],
        interval_type: str,
        dep_var_type: str
    ) -> None:
        """
        Plot response curves for media variables.

        :param dt_mod: Model data
        :param x_decomp_vec: Decomposition vector
        :param x_decomp_vec_immediate: Immediate decomposition vector
        :param x_decomp_vec_carryover: Carryover decomposition vector
        :param all_media: All media variables
        :param paid_media_spends: Paid media spends
        :param interval_type: Type of interval
        :param dep_var_type: Type of dependent variable
        """
        pass

    def plot_media_baseline_contributions(
        self,
        dt_mod: pd.DataFrame,
        x_decomp_vec: pd.DataFrame,
        x_decomp_vec_immediate: pd.DataFrame,
        x_decomp_vec_carryover: pd.DataFrame,
        all_media: List[str],
        paid_media_spends: List[str],
        interval_type: str,
        dep_var_type: str
    ) -> None:
        """
        Plot media and baseline contributions.

        :param dt_mod: Model data
        :param x_decomp_vec: Decomposition vector
        :param x_decomp_vec_immediate: Immediate decomposition vector
        :param x_decomp_vec_carryover: Carryover decomposition vector
        :param all_media: All media variables
        :param paid_media_spends: Paid media spends
        :param interval_type: Type of interval
        :param dep_var_type: Type of dependent variable
        """
        pass

    def plot_spend_share_vs_effect_share(
        self,
        x_decomp_spend_dist: pd.DataFrame,
        x_decomp_spend_dist_immediate: pd.DataFrame,
        x_decomp_spend_dist_carryover: pd.DataFrame,
        paid_media_spends: List[str]
    ) -> None:
        """
        Plot spend share vs effect share for paid media variables.

        :param x_decomp_spend_dist: Decomposition spend distribution
        :param x_decomp_spend_dist_immediate: Immediate decomposition spend distribution
        :param x_decomp_spend_dist_carryover: Carryover decomposition spend distribution
        :param paid_media_spends: Paid media spends
        """
        pass

    def plot_adstock_curves(
        self,
        theta: Dict[str, float],
        shape: Dict[str, float],
        half_life: Dict[str, float],
        all_media: List[str]
    ) -> None:
        """
        Plot adstock curves for media variables.

        :param theta: Theta values for each media variable
        :param shape: Shape values for each media variable
        :param half_life: Half-life values for each media variable
        :param all_media: All media variables
        """
        pass

    def generate_mmm_plots(
        self, 
        plot_pareto: bool = True,
    ) -> None:
        """
        Generate all Marketing Mix Model plots.

        :param plot_pareto: Whether to plot Pareto front
        """
        pass
