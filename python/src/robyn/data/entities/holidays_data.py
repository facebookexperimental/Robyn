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
        prophet_country: str,
        day_interval: int,
    ) -> None:
        """
        Initialize a HolidaysData object.

        Args:
            dt_holidays (pd.DataFrame): A pandas DataFrame containing holiday data.
            prophet_vars (List[ProphetVariableType]): A list of Prophet variable types.
            prophet_signs (List[ProphetSigns]): A list of signs for Prophet variables.
            prophet_country (str): The country for which holidays are defined.
            day_interval (int): The interval between days in the holiday data.

        Returns:
            None
        """
        self.dt_holidays: pd.DataFrame = dt_holidays
        self.prophet_vars: List[ProphetVariableType] = prophet_vars
        self.prophet_signs: List[ProphetSigns] = prophet_signs
        self.prophet_country: str = prophet_country
        self.day_interval: int = day_interval

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
            f"day_interval: {self.day_interval}\n"
        )


    def update(self, **kwargs: object) -> None:
        """
        Update the attributes of the HolidaysData object.
        
        :param kwargs: Keyword arguments corresponding to the attributes to update.
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"{key} is not a valid attribute of HolidaysData")
