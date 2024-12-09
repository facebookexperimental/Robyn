from dataclasses import dataclass
from typing import List, Dict, Optional
import pandas as pd


@dataclass
class ConvergenceData:
    moo_distrb_plot: Optional[str]  # Hexadecimal string of plot image data
    moo_cloud_plot: Optional[str]  # Hexadecimal string of plot image data
    errors: pd.DataFrame
    conv_msg: List[str]

    @classmethod
    def from_dict(cls, data: Dict[str, any]) -> "ConvergenceData":
        return cls(
            moo_distrb_plot=data.get("moo_distrb_plot"),
            moo_cloud_plot=data.get("moo_cloud_plot"),
            errors=pd.DataFrame(data.get("errors", [])),
            conv_msg=data.get("conv_msg", []),
        )

    def to_dict(self) -> Dict[str, any]:
        return {
            "moo_distrb_plot": self.moo_distrb_plot,
            "moo_cloud_plot": self.moo_cloud_plot,
            "errors": self.errors.to_dict(orient="records"),
            "conv_msg": self.conv_msg,
        }
