from dataclasses import dataclass
import pandas as pd
from typing import Dict, Any


@dataclass
class FeaturizedMMMData:
    dt_mod: pd.DataFrame
    dt_modRollWind: pd.DataFrame
    modNLS: Dict[str, Any]
