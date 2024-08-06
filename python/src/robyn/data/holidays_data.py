# pyre-strict

from typing import List, Optional
import pandas as pd
from enums import ProphetVariableType
from enums import ProphetSigns

class HolidaysData:
    def __init__(
        self,
        dt_holidays: pd.DataFrame = None,
        prophet_vars: List[ProphetVariableType] = None,
        prophet_signs: List[ProphetSigns] = None,
        prophet_country: str = None,
        day_interval: int = None,

    ) -> None:
        """
        Initialize a HolidaysData object.

        Args:
        dt_holidays (Optional[pd.DataFrame]): A pandas DataFrame containing holiday data.
        prophet_vars (List[ProphetVariableType]): A list of Prophet variable types.
        prophet_signs (List[ProphetSigns]): A list of signs for Prophet variables.
        prophet_country (Optional[str]): The country for which holidays are defined.
        day_interval (Optional[int]): The interval between days in the holiday data.

        Returns:
        None
        """
        self.dt_holidays: Optional[pd.DataFrame] = dt_holidays
        self.prophet_vars: Optional[List[ProphetVariableType]] = prophet_vars
        self.prophet_signs: Optional[List[ProphetSigns]] = prophet_signs
        self.prophet_country: Optional[str] = prophet_country
        self.day_interval: Optional[int] = day_interval

    def __str__(self) -> str:
        """
        Return a string representation of the HolidaysData object.

        Returns:
        str: A string representation of the object.
        """
        return (
            f"HolidaysData:\n"
            f"dt_holidays: {self.dt_holidays.shape if self.dt_holidays is not None else None}\n"
            f"prophet_vars: {self.prophet_vars}\n"
            f"prophet_signs: {self.prophet_signs}\n"
            f"prophet_country: {self.prophet_country}\n"
            f"day_interval: {self.day_interval}\n"
        )


    def update(self, **kwargs: Optional[str]) -> None:
        """
        Update the attributes of the HolidaysData object.
        
        :param kwargs: Keyword arguments corresponding to the attributes to update.
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"{key} is not a valid attribute of HolidaysData")


# Example usage:
if __name__ == "__main__":
    # Initialize HolidaysData with some example data
    holidays_data: HolidaysData = HolidaysData(
        dt_holidays=pd.DataFrame({"holiday": ["New Year", "Christmas"], "date": ["2023-01-01", "2023-12-25"]}),
        prophet_vars=["holiday"],
        prophet_signs=["positive"],
        prophet_country="US"
    )

    # Print the HolidaysData object
    print(holidays_data)

    # Update some attributes in HolidaysData
    holidays_data.update(prophet_country="UK", prophet_vars=["holiday", "event"])

    # Print the updated HolidaysData object
    print(holidays_data)
