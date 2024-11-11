import pandas as pd
import numpy as np
from typing import List, Dict, Any

from robyn.data.entities.mmmdata import MMMData
from robyn.modeling.entities.modeloutputs import ModelOutputs


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
        self.mmmdata = mmmdata
        self.model_outputs = model_outputs
        self.dt_hyppar = dt_hyppar
        self.dt_coef = dt_coef
        self.media_spend_sorted = media_spend_sorted
        self.select_model = select_model
        self.chn_adstocked = chn_adstocked

    def _get_chn_adstocked_from_output_collect(self) -> pd.DataFrame:
        # Filter the media_vec_collect DataFrame
        chn_adstocked = self.model_outputs.media_vec_collect[
            (self.model_outputs.media_vec_collect["type"] == "adstockedMedia")
            & (self.model_outputs.media_vec_collect["solID"] == self.select_model)
        ]

        # Select only the required media columns
        chn_adstocked = chn_adstocked[self.media_spend_sorted]

        # Slice the DataFrame based on the rolling window
        start_index = self.mmmdata.mmmdata_spec.window_start
        end_index = self.mmmdata.mmmdata_spec.window_end
        chn_adstocked = chn_adstocked.iloc[start_index : end_index + 1]

        return chn_adstocked

    def get_hill_params(self) -> Dict[str, Any]:
        # Extract alphas and gammas from dt_hyppar
        hill_hyp_par_vec = self.dt_hyppar.filter(regex=".*_alphas|.*_gammas").iloc[0]
        alphas = hill_hyp_par_vec[[f"{media}_alphas" for media in self.media_spend_sorted]]
        gammas = hill_hyp_par_vec[[f"{media}_gammas" for media in self.media_spend_sorted]]

        # Handle chn_adstocked
        if self.chn_adstocked is None:
            self.chn_adstocked = self._get_chn_adstocked_from_output_collect()

        # Calculate inflexions
        inflexions = {}
        for i, media in enumerate(self.media_spend_sorted):
            media_range = self.chn_adstocked[media].agg(["min", "max"])
            inflexions[media] = np.dot(media_range, [1 - gammas.iloc[i], gammas.iloc[i]])

        # Get sorted coefficients
        coefs = dict(zip(self.dt_coef["rn"], self.dt_coef["coef"]))
        coefs_sorted = [coefs[media] for media in self.media_spend_sorted]

        return {"alphas": alphas.tolist(), "inflexions": list(inflexions.values()), "coefs_sorted": coefs_sorted}
