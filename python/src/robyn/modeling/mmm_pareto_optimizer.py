# mmm_pareto_optimizer.py

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

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
        if pareto_fronts == "auto":
            pareto_fronts = cls.get_pareto_fronts(modeloutput.trials)

        result_hyp_param = pd.concat(
            [trial.resultCollect.resultHypParam for trial in modeloutput.trials]
        )
        x_decomp_agg = pd.concat(
            [trial.resultCollect.xDecompAgg for trial in modeloutput.trials]
        )

        if calibrated:
            result_calibration = pd.concat(
                [trial.resultCollect.liftCalibration for trial in modeloutput.trials]
            )
            result_calibration = result_calibration.rename(columns={"liftMedia": "rn"})
        else:
            result_calibration = None

        pareto_results = cls.pareto_front(
            x=result_hyp_param["nrmse"],
            y=result_hyp_param["decomp.rssd"],
            fronts=pareto_fronts,
            sort=False,
        )

        result_hyp_param = result_hyp_param.merge(
            pareto_results, left_on=["nrmse", "decomp.rssd"], right_on=["x", "y"]
        )
        result_hyp_param = result_hyp_param.rename(
            columns={"pareto_front": "robynPareto"}
        )
        result_hyp_param = result_hyp_param.sort_values(["iterNG", "iterPar", "nrmse"])

        pareto_solutions = result_hyp_param[
            result_hyp_param["robynPareto"].isin(range(1, pareto_fronts + 1))
        ]["solID"].unique()

        return {
            "pareto_solutions": pareto_solutions,
            "pareto_fronts": pareto_fronts,
            "resultHypParam": result_hyp_param,
            "xDecompAgg": x_decomp_agg,
            "resultCalibration": result_calibration,
        }

    @staticmethod
    def pareto_front(
        x: np.ndarray, y: np.ndarray, fronts: int = 1, sort: bool = True
    ) -> pd.DataFrame:
        points = np.column_stack((x, y))
        fronts_list = []
        for _ in range(fronts):
            pareto = cls.is_pareto_efficient_simple(points)
            fronts_list.append(points[pareto])
            points = points[~pareto]
            if len(points) == 0:
                break

        result = pd.DataFrame(columns=["x", "y", "pareto_front"])
        for i, front in enumerate(fronts_list, 1):
            front_df = pd.DataFrame(front, columns=["x", "y"])
            front_df["pareto_front"] = i
            result = pd.concat([result, front_df])

        if sort:
            result = result.sort_values(["pareto_front", "x", "y"])

        return result

    @staticmethod
    def get_pareto_fronts(pareto_fronts: Union[str, int]) -> int:
        if isinstance(pareto_fronts, str) and pareto_fronts.lower() == "auto":
            return 5  # You might want to implement a more sophisticated logic here
        elif isinstance(pareto_fronts, int):
            return pareto_fronts
        else:
            raise ValueError("pareto_fronts must be 'auto' or an integer")

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
        # Implementation of response calculation
        # This is a placeholder and should be implemented based on your specific requirements
        return pd.DataFrame()

    @staticmethod
    def is_pareto_efficient_simple(costs: np.ndarray) -> np.ndarray:
        is_efficient = np.ones(costs.shape[0], dtype=bool)
        for i, c in enumerate(costs):
            is_efficient[i] = np.all(
                np.any(costs[is_efficient][:i] > c, axis=1)
            ) and np.all(np.any(costs[is_efficient][i + 1 :] > c, axis=1))
        return is_efficient
