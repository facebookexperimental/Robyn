# pyre-strict

from typing import List
import pandas as pd
from dataclasses import dataclass


@dataclass
class ParetoData:
    decomp_spend_dist: pd.DataFrame
    result_hyp_param: pd.DataFrame
    x_decomp_agg: pd.DataFrame
    pareto_fronts: List[int]
