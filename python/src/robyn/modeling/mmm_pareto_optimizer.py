#pyre-strict

from typing import List, Dict, Any, Union, Tuple, Optional
import pandas as pd
import numpy as np

from robyn.data.entities.mmmdata_collection import MMMDataCollection
from robyn.modeling.entities.modeloutput import ModelOutput

class ParetoOptimizer:
    @classmethod
    def pareto_optimize(
        cls,
        mmmdata_collection: MMMDataCollection,
        modeloutput: ModelOutput,
        pareto_fronts: Union[str, int] = "auto",
        min_candidates: int = 100,
        calibration_constraint: float = 0.1,
        calibrated: bool = False,
        **kwargs: Any
    ) -> Dict[str, Any]:
        # Implementation details would go here
        pass

    @staticmethod
    def pareto_front(
        x: np.ndarray,
        y: np.ndarray,
        fronts: int = 1,
        sort: bool = True
    ) -> pd.DataFrame:
        # Implementation details would go here
        pass

    @staticmethod
    def get_pareto_fronts(pareto_fronts: Union[str, int]) -> int:
        # Implementation details would go here
        pass

    @classmethod
    def run_dt_resp(
        cls,
        respN: int,
        mmmdata_collection: MMMDataCollection,
        modeloutput: ModelOutput,
        decompSpendDistPar: pd.DataFrame,
        resultHypParamPar: pd.DataFrame,
        xDecompAggPar: pd.DataFrame,
        **kwargs: Any
    ) -> pd.DataFrame:
        # Implementation details would go here
        pass