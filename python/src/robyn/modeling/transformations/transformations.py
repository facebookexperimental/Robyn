# pyre-strict

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from robyn.data.entities.enums import AdstockType
from robyn.data.entities.hyperparameters import ChannelHyperparameters, Hyperparameters
from robyn.data.entities.mmmdata import MMMData
from robyn.modeling.feature_engineering import FeaturizedMMMData
from scipy import stats


@dataclass
class AdstockResult:
    x: pd.Series
    x_decayed: pd.Series
    x_imme: pd.Series
    thetaVec: Optional[pd.Series]
    thetaVecCum: pd.Series
    inflation_total: float


@dataclass
class TransformationResult:
    dt_modSaturated: pd.DataFrame
    dt_saturatedImmediate: pd.DataFrame
    dt_saturatedCarryover: pd.DataFrame


class Transformation:
    def __init__(self, mmm_data: MMMData):
        """
        Initialize the Transformation class with MMMData.

        Args:
            mmm_data (MMMData): The MMMData object containing all input data.
        """
        self.mmm_data = mmm_data

    def normalize(x: pd.Series) -> pd.Series:
        """
        Normalize a series to [0, 1] range.

        Args:
            x (pd.Series): The series to normalize.

        Returns:
            pd.Series: The normalized series.
        """
        if x.max() == x.min():
            return pd.Series([1] + [0] * (len(x) - 1), index=x.index)
        return (x - x.min()) / (x.max() - x.min())

    def mic_men(
        x: Union[float, pd.Series], Vmax: float, Km: float, reverse: bool = False
    ) -> Union[float, pd.Series]:
        """
        Apply Michaelis-Menten transformation.

        Args:
            x (Union[float, pd.Series]): Input value(s).
            Vmax (float): Maximum rate achieved by the system.
            Km (float): Michaelis constant.
            reverse (bool, optional): If True, apply reverse transformation. Defaults to False.

        Returns:
            Union[float, pd.Series]: Transformed value(s).
        """
        if not reverse:
            return Vmax * x / (Km + x)
        else:
            return x * Km / (Vmax - x)

    def adstock_geometric(self, x: pd.Series, theta: float) -> AdstockResult:
        """
        Apply geometric adstock transformation.

        Args:
            x (pd.Series): Input series.
            theta (float): Decay rate.

        Returns:
            AdstockResult: Result of the adstock transformation.
        """
        if not np.isscalar(theta):
            raise ValueError("theta must be a single value")
        x_decayed = pd.Series(index=x.index, dtype=float)
        x_decayed.iloc[0] = x.iloc[0]
        for xi in range(1, len(x_decayed)):
            x_decayed.iloc[xi] = x.iloc[xi] + theta * x_decayed.iloc[xi - 1]
        thetaVecCum = pd.Series(theta ** np.arange(len(x)), index=x.index)
        inflation_total = x_decayed.sum() / x.sum()
        return AdstockResult(x, x_decayed, x, None, thetaVecCum, inflation_total)

    def adstock_weibull(
        self,
        x: pd.Series,
        shape: float,
        scale: float,
        windlen: Optional[int] = None,
        adstockType: AdstockType = AdstockType.WEIBULL_CDF,
    ) -> AdstockResult:
        """
        Apply Weibull adstock transformation.

        Args:
            x (pd.Series): Input series.
            shape (float): Shape parameter of the Weibull distribution.
            scale (float): Scale parameter of the Weibull distribution.
            windlen (Optional[int], optional): Window length. Defaults to None.
            adstockType (AdstockType, optional): Type of Weibull function to use. Defaults to AdstockType.WEIBULL_CDF.

        Returns:
            AdstockResult: Result of the adstock transformation.
        """
        if not (np.isscalar(shape) and np.isscalar(scale)):
            raise ValueError("shape and scale must be single values")
        windlen = len(x) if windlen is None else windlen

        x_bin = pd.Series(range(1, windlen + 1))
        scaleTrans = int(x_bin.quantile(scale))

        if shape == 0 or scale == 0:
            x_decayed = x
            thetaVecCum = thetaVec = pd.Series(np.zeros(windlen), index=x_bin)
            x_imme = x
        else:
            if adstockType == AdstockType.WEIBULL_CDF:
                thetaVec = pd.Series(
                    [1]
                    + list(
                        1 - stats.weibull_min.cdf(x_bin[:-1], shape, scale=scaleTrans)
                    ),
                    index=x_bin,
                )
                thetaVecCum = thetaVec.cumprod()
            elif adstockType == AdstockType.WEIBULL_PDF:
                thetaVecCum = normalize(
                    pd.Series(
                        stats.weibull_min.pdf(x_bin, shape, scale=scaleTrans),
                        index=x_bin,
                    )
                )

            x_decayed = pd.Series(0, index=x.index, dtype=float)
            for i, xi in enumerate(x):
                x_decayed += pd.Series(
                    np.convolve(
                        np.concatenate((np.zeros(i), np.repeat(xi, windlen - i))),
                        thetaVecCum,
                    )[:windlen],
                    index=x.index,
                )

            x_imme = (
                x
                if adstockType == AdstockType.WEIBULL_CDF
                else pd.Series(np.convolve(x, thetaVecCum)[: len(x)], index=x.index)
            )

        inflation_total = x_decayed.sum() / x.sum()
        return AdstockResult(
            x,
            x_decayed,
            x_imme,
            thetaVec if "thetaVec" in locals() else None,
            thetaVecCum,
            inflation_total,
        )

    def transform_adstock(
        self,
        x: pd.Series,
        adstockType: AdstockType,
        channelHyperparameters: ChannelHyperparameters,
        windlen: Optional[int] = None,
    ) -> AdstockResult:
        """
        Apply adstock transformation based on the specified method.

        Args:
            x (pd.Series): Input series.
            adstockType (AdstockType): Adstock method to use.
            theta (Optional[float], optional): Decay rate for geometric adstock. Defaults to None.
            shape (Optional[float], optional): Shape parameter for Weibull adstock. Defaults to None.
            scale (Optional[float], optional): Scale parameter for Weibull adstock. Defaults to None.
            windlen (Optional[int], optional): Window length for Weibull adstock. Defaults to None.

        Returns:
            AdstockResult: Result of the adstock transformation.
        """
        if adstockType == AdstockType.GEOMETRIC:
            theta = channelHyperparameters.thetas[0]
            return self.adstock_geometric(x, theta)
        elif adstockType in [AdstockType.WEIBULL_CDF, AdstockType.WEIBULL_PDF]:
            shape = channelHyperparameters.shapes[0]
            scale = channelHyperparameters.scales[0]
            return self.adstock_weibull(self, x, shape, scale, windlen, adstockType)

    def saturation_hill(
        self,
        x: pd.Series,
        alpha: float,
        gamma: float,
        x_marginal: Optional[pd.Series] = None,
    ) -> pd.Series:
        """
        Apply Hill function for saturation.

        Args:
            x (pd.Series): Input series.
            alpha (float): Alpha parameter controlling the shape of the saturation curve.
            gamma (float): Gamma parameter controlling the inflexion point of the saturation curve.
            x_marginal (Optional[pd.Series], optional): Marginal input series. Defaults to None.

        Returns:
            pd.Series: Saturated series.
        """
        if not (np.isscalar(alpha) and np.isscalar(gamma)):
            raise ValueError("alpha and gamma must be single values")

        # Calculate the inflection point
        inflexion = gamma * x.max() + (1 - gamma) * x.min()

        if x_marginal is None:
            # Saturation curve without marginal input
            return x**alpha / (x**alpha + inflexion**alpha)
        else:
            # Saturation curve with marginal input
            return x_marginal**alpha / (x_marginal**alpha + inflexion**alpha)

    def plot_adstock(self, plot: bool = True) -> Optional[plt.Figure]:
        """
        Generate adstock comparison plots.

        Args:
            plot (bool, optional): Whether to generate the plot. Defaults to True.

        Returns:
            Optional[plt.Figure]: The generated figure if plot is True, None otherwise.
        """
        if plot:
            channelHyperparameters = ChannelHyperparameters(
                shapes=[0.5, 1, 2, 3, 5], scales=[0.5], thetas=[0.1, 0.3, 0.5, 0.7, 0.9]
            )
            x = pd.Series(range(1, 101))
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

            # Geometric adstock
            for theta in channelHyperparameters.thetas:
                y = self.adstock_geometric(x, theta).thetaVecCum
                ax1.plot(x, y, label=f"theta = {theta}")
            ax1.set_title("Geometric Adstock\n(Fixed decay rate)")
            ax1.set_xlabel("Time unit")
            ax1.set_ylabel("Media decay")
            ax1.legend()

            # Weibull adstock
            for shape in channelHyperparameters.shapes:
                for scale in channelHyperparameters.scales:
                    y_cdf = self.adstock_weibull(
                        self, x, shape, scale, adstockType=AstockType.WEIBULL_CDF
                    ).thetaVecCum
                    y_pdf = self.adstock_weibull(
                        self, x, shape, scale, adstockType=AstockType.WEIBULL_PDF
                    ).thetaVecCum
                    ax2.plot(x, y_cdf, label=f"CDF: shape={shape}, scale={scale}")
                    ax2.plot(
                        x,
                        y_pdf,
                        linestyle="--",
                        label=f"PDF: shape={shape}, scale={scale}",
                    )
            ax2.set_title("Weibull Adstock CDF vs PDF\n(Flexible decay rate)")
            ax2.set_xlabel("Time unit")
            ax2.set_ylabel("Media decay accumulated")
            ax2.legend()

            plt.tight_layout()
            return fig
        return None

    def plot_saturation(plot: bool = True) -> Optional[plt.Figure]:
        """
        Generate saturation comparison plots.

        Args:
            plot (bool, optional): Whether to generate the plot. Defaults to True.

        Returns:
            Optional[plt.Figure]: The generated figure if plot is True, None otherwise.
        """
        figures = []  # List to hold figure objects
        if plot:
            x_sample = np.arange(1, 101)  # Sample from 1 to 100
            alpha_samp = [0.1, 0.5, 1, 2, 3]
            gamma_samp = [0.1, 0.3, 0.5, 0.7, 0.9]

            # Plot for varying alpha
            hill_alpha_collect = []
            for alpha in alpha_samp:
                y = x_sample**alpha / (x_sample**alpha + (0.5 * 100) ** alpha)
                hill_alpha_collect.append(
                    pd.DataFrame({"x": x_sample, "y": y, "alpha": alpha})
                )

            hill_alpha_collect = pd.concat(hill_alpha_collect)
            hill_alpha_collect["alpha"] = hill_alpha_collect["alpha"].astype("category")

            plt.figure(figsize=(10, 6))
            sns.lineplot(
                data=hill_alpha_collect, x="x", y="y", hue="alpha", palette="Set2"
            )
            plt.title("Cost response with hill function")
            plt.suptitle("Alpha changes while gamma = 0.5", y=0.95)
            plt.xlabel("x")
            plt.ylabel("y")
            plt.grid()
            figures.append(plt.gcf())  # Store the current figure
            plt.show()

            # Plot for varying gamma
            hill_gamma_collect = []
            for gamma in gamma_samp:
                y = x_sample**2 / (x_sample**2 + (gamma * 100) ** 2)
                hill_gamma_collect.append(
                    pd.DataFrame({"x": x_sample, "y": y, "gamma": gamma})
                )

            hill_gamma_collect = pd.concat(hill_gamma_collect)
            hill_gamma_collect["gamma"] = hill_gamma_collect["gamma"].astype("category")

            plt.figure(figsize=(10, 6))
            sns.lineplot(
                data=hill_gamma_collect, x="x", y="y", hue="gamma", palette="Set2"
            )
            plt.title("Cost response with hill function")
            plt.suptitle("Gamma changes while alpha = 2", y=0.95)
            plt.xlabel("x")
            plt.ylabel("y")
            plt.grid()
            figures.append(plt.gcf())  # Store the current figure
            plt.show()

        return figures  # Return the list of figure objects

    def run_transformations(
        self,
        featurized_data: FeaturizedMMMData,
        hyperparameters: Hyperparameters,
        adstockType: AdstockType,  # Use AdstockType enum
        **kwargs,
    ) -> TransformationResult:
        """
        Run media transformations including adstock and saturation.

        Args:
            featurized_data (FeaturizedMMMData): The featurized data containing modified data.
            hyperparameters (Hyperparameters): Hyperparameters for the transformations.
            adstockType (AdstockType): The type of adstock to apply.
            **kwargs: Additional keyword arguments.

        Returns:
            TransformationResult: Result of the transformations.
        """
        all_media: List[str] = self.mmm_data.mmmdata_spec.all_media
        rollingWindowStartWhich: int = (
            self.mmm_data.mmmdata_spec.rolling_window_start_which
        )
        rollingWindowEndWhich: int = self.mmm_data.mmmdata_spec.rolling_window_end_which
        dt_modAdstocked: pd.DataFrame = featurized_data.dt_mod.copy()

        # Handle 'ds' column
        if "ds" in dt_modAdstocked.columns:
            dt_modAdstocked.set_index("ds", inplace=True)

        mediaAdstocked: Dict[str, pd.Series] = {}
        mediaSaturated: Dict[str, pd.Series] = {}
        mediaSaturatedImmediate: Dict[str, pd.Series] = {}
        mediaSaturatedCarryover: Dict[str, pd.Series] = {}

        for media in all_media:
            m: pd.Series = dt_modAdstocked[media]
            channelHyperparameters: ChannelHyperparameters = (
                hyperparameters.get_hyperparameter(media)
            )
            if adstockType == AdstockType.GEOMETRIC:
                x_list: AdstockResult = self.transform_adstock(
                    m, adstockType, channelHyperparameters
                )
            elif adstockType in [AdstockType.WEIBULL_CDF, AdstockType.WEIBULL_PDF]:
                x_list: AdstockResult = self.transform_adstock(
                    m, adstockType, channelHyperparameters
                )

            m_imme: pd.Series = (
                x_list.x_imme if adstockType == AdstockType.WEIBULL_PDF else m
            )
            m_adstocked: pd.Series = x_list.x_decayed
            mediaAdstocked[media] = m_adstocked
            m_carryover: pd.Series = m_adstocked - m_imme

            m_adstockedRollWind: pd.Series = m_adstocked.iloc[
                rollingWindowStartWhich:rollingWindowEndWhich
            ]
            m_carryoverRollWind: pd.Series = m_carryover.iloc[
                rollingWindowStartWhich:rollingWindowEndWhich
            ]

            mediaSaturated[media] = self.saturation_hill(
                m_adstockedRollWind,
                channelHyperparameters.alphas[0],
                channelHyperparameters.gammas[0],
            )
            mediaSaturatedCarryover[media] = self.saturation_hill(
                m_adstockedRollWind,
                channelHyperparameters.alphas[0],
                channelHyperparameters.gammas[0],
                x_marginal=m_carryoverRollWind,
            )
            mediaSaturatedImmediate[media] = (
                mediaSaturated[media] - mediaSaturatedCarryover[media]
            )

        # Combine transformed data
        dt_modAdstocked = pd.concat(
            [dt_modAdstocked.drop(columns=all_media), pd.DataFrame(mediaAdstocked)],
            axis=1,
        )
        dt_modSaturated = pd.concat(
            [
                dt_modAdstocked.iloc[
                    rollingWindowStartWhich:rollingWindowEndWhich
                ].drop(columns=all_media),
                pd.DataFrame(mediaSaturated),
            ],
            axis=1,
        )
        dt_saturatedImmediate = pd.DataFrame(mediaSaturatedImmediate).fillna(0)
        dt_saturatedCarryover = pd.DataFrame(mediaSaturatedCarryover).fillna(0)

        return TransformationResult(
            dt_modSaturated, dt_saturatedImmediate, dt_saturatedCarryover
        )
