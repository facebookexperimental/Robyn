import pandas as pd
from dataclasses import dataclass


@dataclass
class ConfidenceIntervalCollectionData:
    confidence_interval_df: pd.DataFrame
    sim_collect: pd.DataFrame
    boot_n: int
    sim_n: int
