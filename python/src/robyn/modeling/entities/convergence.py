from dataclasses import dataclass
from typing import List, Dict, Any
import pandas as pd


@dataclass
class Convergence:
    moo_distrb_plot: Any  # This could be a more specific type depending on how you want to store the plot data
    moo_cloud_plot: Any  # This could be a more specific type depending on how you want to store the plot data
    errors: pd.DataFrame
    conv_msg: List[str]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Convergence":
        return cls(
            moo_distrb_plot=data.get("moo_distrb_plot"),
            moo_cloud_plot=data.get("moo_cloud_plot"),
            errors=pd.DataFrame(data.get("errors", [])),
            conv_msg=data.get("conv_msg", []),
        )
