from dataclasses import dataclass
from typing import Dict, Optional
import pandas as pd

@dataclass
class PlotData:
    spend_data: pd.DataFrame
    response_data: pd.DataFrame
    model_metrics: Dict[str, float]
    adstock_data: pd.DataFrame
    response_curves: pd.DataFrame
    fitted_vs_actual: pd.DataFrame
    diagnostic_data: pd.DataFrame
    carryover_data: pd.DataFrame
    bootstrap_data: Optional[pd.DataFrame] = None