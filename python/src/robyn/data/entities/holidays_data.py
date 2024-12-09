# pyre-strict

from typing import List
import pandas as pd
from robyn.data.entities.enums import ProphetVariableType
from robyn.data.entities.enums import ProphetSigns


class HolidaysData:
    def __init__(
        self,
        dt_holidays: pd.DataFrame,
        prophet_vars: List[ProphetVariableType],
        prophet_signs: List[ProphetSigns],
        prophet_country: str,  # TODO Is there a library for country codes so that we can type
    ) -> None:
        """
        Initialize a HolidaysData object.

        Args:
            dt_holidays (pd.DataFrame): A pandas DataFrame containing holiday data.
            prophet_vars (List[ProphetVariableType]): A list of Prophet variable types.
            prophet_signs (List[ProphetSigns]): A list of signs for Prophet variables.
            prophet_country (str): The country for which holidays are defined.

        Returns:
            None
        """
        self.dt_holidays: pd.DataFrame = dt_holidays
        self.prophet_vars: List[ProphetVariableType] = prophet_vars
        self.prophet_signs: List[ProphetSigns] = prophet_signs
        self.prophet_country: str = prophet_country

    def __str__(self) -> str:
        """
        Return a string representation of the HolidaysData object.

        Returns:
        str: A string representation of the object.
        """
        return (
            f"HolidaysData:\n"
            f"dt_holidays: {self.dt_holidays.shape}\n"
            f"prophet_vars: {self.prophet_vars}\n"
            f"prophet_signs: {self.prophet_signs}\n"
            f"prophet_country: {self.prophet_country}\n"
        )
