import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, Any

class DataSource(ABC):
    @abstractmethod
    def ingest(self, start_date: str, end_date: str) -> pd.DataFrame:
        pass
