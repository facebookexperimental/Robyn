# pyre-strict

from typing import List, Optional
import pandas as pd

class HolidaysData:
    def __init__(
        self,
        dt_holidays: Optional[pd.DataFrame] = None,
        prophet_vars: Optional[List[str]] = None,
        prophet_signs: Optional[List[str]] = None,
        prophet_country: Optional[str] = None
    ) -> None:
        self.dt_holidays: Optional[pd.DataFrame] = dt_holidays
        self.prophet_vars: Optional[List[str]] = prophet_vars
        self.prophet_signs: Optional[List[str]] = prophet_signs
        self.prophet_country: Optional[str] = prophet_country

    def __str__(self) -> str:
        return (
            f"HolidaysData:\n"
            f"dt_holidays: {self.dt_holidays.shape if self.dt_holidays is not None else None}\n"
            f"prophet_vars: {self.prophet_vars}\n"
            f"prophet_signs: {self.prophet_signs}\n"
            f"prophet_country: {self.prophet_country}\n"
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
