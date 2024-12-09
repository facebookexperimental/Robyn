import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any

from robyn.data.entities.mmmdata import MMMData
from robyn.modeling.entities.modeloutputs import ModelOutputs

logger = logging.getLogger(__name__)


class HillCalculator:
    def __init__(
        self,
        mmmdata: MMMData,
        model_outputs: ModelOutputs,
        dt_hyppar: pd.DataFrame,
        dt_coef: pd.DataFrame,
        media_spend_sorted: List[str],
        select_model: str,
        chn_adstocked: pd.DataFrame = None,
    ):
        logger.debug(
            "Initializing HillCalculator with parameters: MMMData=%s, ModelOutputs=%s, "
            "media_spend_sorted=%s, select_model=%s",
            mmmdata,
            model_outputs,
            media_spend_sorted,
            select_model,
        )

        self.mmmdata = mmmdata
        self.model_outputs = model_outputs
        self.dt_hyppar = dt_hyppar
        self.dt_coef = dt_coef
        self.media_spend_sorted = media_spend_sorted
        self.select_model = select_model
        self.chn_adstocked = chn_adstocked

        logger.debug("HillCalculator initialized successfully")

    def _get_chn_adstocked_from_output_collect(self) -> pd.DataFrame:
        logger.debug("Retrieving channel adstocked data from output collect")
        try:
            # Filter the media_vec_collect DataFrame
            logger.debug(
                "Filtering media_vec_collect for adstockedMedia and solID=%s",
                self.select_model,
            )
            chn_adstocked = self.model_outputs.media_vec_collect[
                (self.model_outputs.media_vec_collect["type"] == "adstockedMedia")
                & (self.model_outputs.media_vec_collect["solID"] == self.select_model)
            ]

            if chn_adstocked.empty:
                logger.warning(
                    "No adstocked media data found for solID=%s", self.select_model
                )
                return pd.DataFrame()

            # Select only the required media columns
            logger.debug("Selecting media columns: %s", self.media_spend_sorted)
            chn_adstocked = chn_adstocked[self.media_spend_sorted]

            # Slice the DataFrame based on the rolling window
            start_index = self.mmmdata.mmmdata_spec.window_start
            end_index = self.mmmdata.mmmdata_spec.window_end
            logger.debug(
                "Slicing DataFrame with window: start=%d, end=%d",
                start_index,
                end_index,
            )

            chn_adstocked = chn_adstocked.iloc[start_index : end_index + 1]

            logger.debug(
                "Successfully retrieved channel adstocked data with shape %s",
                chn_adstocked.shape,
            )
            return chn_adstocked

        except Exception as e:
            logger.error("Error retrieving channel adstocked data: %s", str(e))
            raise

    def get_hill_params(self) -> Dict[str, Any]:
        logger.debug("Calculating Hill parameters")
        try:
            # Extract alphas and gammas from dt_hyppar
            logger.debug("Extracting alphas and gammas from hyperparameters")
            hill_hyp_par_vec = self.dt_hyppar.filter(regex=".*_alphas|.*_gammas").iloc[
                0
            ]
            alphas = hill_hyp_par_vec[
                [f"{media}_alphas" for media in self.media_spend_sorted]
            ]
            gammas = hill_hyp_par_vec[
                [f"{media}_gammas" for media in self.media_spend_sorted]
            ]

            logger.debug(
                "Extracted parameters - alphas: %s, gammas: %s",
                alphas.to_dict(),
                gammas.to_dict(),
            )

            # Handle chn_adstocked
            if self.chn_adstocked is None:
                logger.debug(
                    "No pre-calculated channel adstocked data found, retrieving from output collect"
                )
                self.chn_adstocked = self._get_chn_adstocked_from_output_collect()

            # Calculate inflexions
            logger.debug("Calculating inflexion points for each media channel")
            inflexions = {}
            for i, media in enumerate(self.media_spend_sorted):
                media_range = self.chn_adstocked[media].agg(["min", "max"])
                inflexion = np.dot(media_range, [1 - gammas.iloc[i], gammas.iloc[i]])
                inflexions[media] = inflexion
                logger.debug("Calculated inflexion point for %s: %f", media, inflexion)

            # Get sorted coefficients
            logger.debug("Sorting coefficients according to media spend order")
            coefs = dict(zip(self.dt_coef["rn"], self.dt_coef["coef"]))
            coefs_sorted = [coefs[media] for media in self.media_spend_sorted]

            result = {
                "alphas": alphas.tolist(),
                "inflexions": list(inflexions.values()),
                "coefs_sorted": coefs_sorted,
            }

            logger.debug("Successfully calculated Hill parameters")
            logger.debug("Final Hill parameters: %s", result)

            return result

        except Exception as e:
            logger.error("Error calculating Hill parameters: %s", str(e))
            raise
