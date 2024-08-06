from typing import List, Dict, Union, Optional
import numpy as np
import pandas as pd
from scipy.stats import weibull_min

class AdstockSaturationTransformation:
    def __init__(self) -> None:
        pass

    def mic_men(self, substrate_concentration: float, Vmax: float, Km: float, reverse: bool = False) -> float:
        """
        Calculate the Michaelis-Menten transformation.

        Args:
            substrate_concentration: The concentration of the substrate.
            Vmax: The maximum rate of the transformation.
            Km: The Michaelis constant.
            reverse: Whether to perform the reverse transformation.

        Returns:
            The transformed value.
        """
        pass

    def adstock_geometric(self, advertising_spend: List[float], theta: float) -> Dict[str, Union[List[float], float]]:
        """
        Adstock geometric function

        Args:
            advertising_spend: The list of advertising spend values.
            theta: The decay factor.

        Returns:
            A dictionary containing the original input values, the decayed values,
            the cumulative decay factors, and the inflation total.
        """
        pass

    def adstock_weibull(self, media_impressions: List[float], shape: float, scale: float, windlen: Optional[int] = None, stype: str = "cdf") -> Dict[str, Union[List[float], float, np.ndarray]]:
        """
        Adstock Weibull function

        Args:
            media_impressions: The time series data of media impressions (e.g. TV, digital, etc.).
            shape: The shape parameter of the Weibull distribution.
            scale: The scale parameter of the Weibull distribution.
            windlen: The length of the adstock window.
            stype: The type of adstock transformation to perform.

        Returns:
            A dictionary containing the transformed data and related information.
        """
        pass
    # Used AI to generate the following code by giving reference to R code.
    # def adstock_weibull(self, media_impressions: List[float], shape: float, scale: float, windlen: Optional[int] = None, stype: str = "cdf") -> Dict[str, Union[List[float], float, np.ndarray]]:
    #     """
    #     Adstock Weibull function

    #     Args:
    #         media_impressions: The time series data of media impressions (e.g. TV, digital, etc.).
    #         shape: The shape parameter of the Weibull distribution.
    #         scale: The scale parameter of the Weibull distribution.
    #         windlen: The length of the adstock window.
    #         stype: The type of adstock transformation to perform.

    #     Returns:
    #         A dictionary containing the transformed data and related information.
    #     """
    #     media_impressions = np.array(media_impressions)

    #     if windlen is None:
    #         windlen = len(media_impressions)

    #     # Check if shape and scale are single values
    #     if not np.isscalar(shape) or not np.isscalar(scale):
    #         raise ValueError("Shape and scale must be single values")

    #     # Check if stype is valid
    #     if stype.lower() not in ["cdf", "pdf"]:
    #         raise ValueError("Invalid stype. Must be 'cdf' or 'pdf'")

    #     if len(media_impressions) > 1:
    #         media_impressions_bin = np.arange(1, windlen + 1)
    #         scale_trans = np.round(np.quantile(media_impressions_bin, scale))

    #         if shape == 0 or scale == 0:
    #             media_impressions_decayed = media_impressions
    #             theta_vec_cum = np.zeros(windlen)
    #             media_impressions_imme = media_impressions
    #         else:
    #             if stype.lower() == "cdf":
    #                 theta_vec = np.concatenate(([1], 1 - weibull_min.cdf(media_impressions_bin[:-1], shape, scale=scale_trans)))
    #                 theta_vec_cum = np.cumprod(theta_vec)
    #             elif stype.lower() == "pdf":
    #                 theta_vec_cum = weibull_min.pdf(media_impressions_bin, shape, scale=scale_trans) / np.sum(weibull_min.pdf(media_impressions_bin, shape, scale=scale_trans))

    #             media_impressions_decayed = np.array([
    #                 np.sum(np.concatenate((np.zeros(i), np.repeat(media_impression, windlen - i))) * np.roll(theta_vec_cum, i))
    #                 for i, media_impression in enumerate(media_impressions)
    #             ])

    #             media_impressions_imme = np.diag(np.array([
    #                 np.concatenate((np.zeros(i), np.repeat(media_impression, windlen - i))) * np.roll(theta_vec_cum, i)
    #                 for i, media_impression in enumerate(media_impressions)
    #             ]))

    #     else:
    #         media_impressions_decayed = media_impressions_imme = media_impressions
    #         theta_vec_cum = np.array([1])

    #     inflation_total = np.sum(media_impressions_decayed) / np.sum(media_impressions)

    #     return {
    #         "media_impressions": media_impressions.tolist(),
    #         "media_impressions_decayed": media_impressions_decayed.tolist(),
    #         "theta_vec_cum": theta_vec_cum.tolist(),
    #         "inflation_total": inflation_total,
    #         "media_impressions_imme": media_impressions_imme.tolist()
    #     }   

    def transform_adstock(self, media_spend_data: List[float], adstock: Literal["geometric", "weibull_cdf", "weibull_pdf"], theta: Optional[float] = None, shape: Optional[float] = None, scale: Optional[float] = None, windlen: Optional[int] = None) -> Dict[str, Union[List[float], float, np.ndarray]]:
        """
        Transforms the input data using the adstock model.

        Args:
            media_spend_data: The time series data of media spend (e.g. advertising spend, impressions, etc.).
            adstock: The type of adstock model to be applied.
            theta: The decay factor for the geometric adstock model.
            shape: The shape parameter for the Weibull adstock model.
            scale: The scale parameter for the Weibull adstock model.
            windlen: The length of the adstock window.

        Returns:
            The transformed data based on the adstock model.
        """
        pass

    #TODO: Do we need this method? What data is it normalizing?
    def normalize(self, x: np.ndarray) -> np.ndarray:
        """
        Normalizes the input data.

        Args:
            x: The input data to be normalized.

        Returns:
            The normalized data.
        """
        pass

    def saturation_hill(self, advertising_intensities: np.ndarray, response_sensitivity: float, saturation_rate: float, baseline_intensities: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Implements the saturation hill function.

        Args:
            advertising_intensities: Input values representing the intensity of advertising efforts.
            response_sensitivity: Exponent parameter controlling the responsiveness to advertising.
            saturation_rate: Weighting parameter controlling the rate of saturation.
            baseline_intensities: Optional marginal values representing the baseline advertising intensity.

        Returns:
            Output values computed using the saturation hill function, representing the saturated response to advertising.
        """
        pass
    
    #TODO method should take only required input data instead of MMMDataCollection. Send only information that is needed.
    #TODO review return type. It is open ended dictionary. Can make strictly typed dictionary
    def run_transformations(data_collection: MMMDataCollection, hyperparameters: Dict[str, HyperParameterConfig]) -> Dict[str, pd.DataFrame]:
        """
        Transform media for model fitting.

        Args:
        - data_collection (MMMDataCollection): A collection of data for the media.
        - hyperparameters (HyperParameterConfig): A configuration of hyperparameters.

        Returns:
        - A dictionary containing the transformed data.
        """
        pass
