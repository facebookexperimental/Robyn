# pyre-strict

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Any
import pandas as pd
from robyn.data.entities.enums import (
    ContextSigns,
    DependentVarType,
    OrganicSigns,
    PaidMediaSigns,
)


@dataclass
class MMMData:
    class MMMDataSpec:
        """
        Dependent Variable (Target Variable)
            dep_var: The name of the column in the input dataframe that represents the dependent variable (target variable) that we want to model. This is the variable that we're trying to predict or explain. For example, it could be "sales", "revenue", "conversions", etc.
            dep_var_type: The type of the dependent variable. In this case, it's an enumeration (DependentVarType) that can take values like REVENUE, CONVERSIONS, etc. This helps the model understand the nature of the dependent variable.
        Date Variable
            date_var: The name of the column in the input dataframe that represents the date variable. This is used to specify the time period for which the data is collected. If set to "auto", the model will automatically detect the date column.
        Paid Media Variables
            paid_media_spends: A list of column names in the input dataframe that represent the paid media spends (e.g., advertising expenses). These variables are used to model the impact of paid media on the dependent variable.
            paid_media_vars: A list of column names in the input dataframe that represent additional paid media variables (e.g., ad impressions, clicks, etc.). These variables can be used to model non-linear relationships between paid media and the dependent variable.
            paid_media_signs: A list of signs (positive or negative) that indicate the expected direction of the relationship between each paid media variable and the dependent variable.
        Organic Variables
            organic_vars: A list of column names in the input dataframe that represent organic variables (e.g., social media engagement, content metrics, etc.). These variables are used to model the impact of organic factors on the dependent variable.
            organic_signs: A list of signs (positive or negative) that indicate the expected direction of the relationship between each organic variable and the dependent variable.
        Context Variables
            context_vars: A list of column names in the input dataframe that represent context variables (e.g., seasonality, weather, economic indicators, etc.). These variables are used to model external factors that can impact the dependent variable.
            context_signs: A list of signs (positive or negative) that indicate the expected direction of the relationship between each context variable and the dependent variable.
        Factor Variables
            factor_vars: A list of column names in the input dataframe that represent factor variables (e.g., categorical variables like region, product category, etc.). These variables can be used to model non-linear relationships and interactions between variables.

        """

        def __init__(
            self,
            dep_var: Optional[str] = None,
            dep_var_type: DependentVarType = DependentVarType.REVENUE,
            date_var: str = "auto",
            window_start: datetime = None,
            window_end: datetime = None,
            rolling_window_length: int = None,
            rolling_window_start_which: int = 0,
            rolling_window_end_which: int = None,
            paid_media_spends: Optional[List[str]] = None,
            paid_media_vars: Optional[List[str]] = None,
            paid_media_signs: Optional[List[PaidMediaSigns]] = None,
            organic_vars: Optional[List[str]] = None,
            organic_signs: Optional[List[OrganicSigns]] = None,
            context_vars: Optional[List[str]] = None,
            context_signs: Optional[List[ContextSigns]] = None,
            factor_vars: Optional[List[str]] = None,
            all_media: Optional[List[str]] = None,
            day_interval: Optional[int] = 7,
            interval_type: Optional[str] = "week",
        ) -> None:
            self.dep_var: Optional[str] = dep_var
            self.dep_var_type: DependentVarType = dep_var_type
            self.date_var: str = date_var
            self.window_start: datetime = window_start
            self.window_end: datetime = window_end
            self.rolling_window_length: int = rolling_window_length
            self.rolling_window_start_which: int = rolling_window_start_which
            self.rolling_window_end_which: int = rolling_window_end_which
            self.paid_media_spends: Optional[List[str]] = paid_media_spends
            self.paid_media_vars: Optional[List[str]] = paid_media_vars
            self.paid_media_signs: Optional[List[str]] = paid_media_signs
            self.organic_vars: Optional[List[str]] = organic_vars
            self.organic_signs: Optional[List[str]] = organic_signs
            self.context_vars: Optional[List[str]] = context_vars
            self.context_signs: Optional[List[str]] = context_signs
            self.factor_vars: Optional[List[str]] = factor_vars
            self.all_media = all_media or (
                paid_media_spends + organic_vars if organic_vars else paid_media_spends
            )
            self.day_interval: Optional[int] = day_interval
            self.interval_type: Optional[str] = interval_type

        def __str__(self) -> str:
            return f"""
            MMMDataSpec:
            dep_var: {self.dep_var}
            dep_var_type: {self.dep_var_type}
            date_var: {self.date_var}
            window_start: {self.window_start}
            window_end: {self.window_end}
            rolling_window_length: {self.rolling_window_length}
            rolling_window_start_which: {self.rolling_window_start_which}
            rolling_window_end_which: {self.rolling_window_end_which}
            paid_media_spends: {self.paid_media_spends}
            paid_media_vars: {self.paid_media_vars}
            paid_media_signs: {self.paid_media_signs}
            organic_vars: {self.organic_vars}
            organic_signs: {self.organic_signs}
            context_vars: {self.context_vars}
            context_signs: {self.context_signs}
            factor_vars: {self.factor_vars}
            all_media: {self.all_media}
            day_interval: {self.day_interval}
            interval_type: {self.interval_type}
            """

        def update(self, **kwargs: Any) -> None:
            """
            Update the attributes of the MMMDataSpec object.

            :param kwargs: Keyword arguments corresponding to the attributes to update.
            """
            for key, value in kwargs.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                else:
                    raise AttributeError(
                        f"{key} is not a valid attribute of MMMDataSpec"
                    )

    def __init__(self, data: pd.DataFrame, mmmdata_spec: MMMDataSpec) -> None:
        """
        Initialize the MMMData class with a pandas DataFrame and an MMMDataSpec object.

        :param data: A pandas DataFrame containing the data.
        :param mmmdata_spec: An MMMDataSpec object containing mapping of what is what in the provided data.
        """
        self.data: pd.DataFrame = data
        self.mmmdata_spec: MMMData.MMMDataSpec = mmmdata_spec
        self.calculate_rolling_window_indices()

    def __str__(self) -> str:
        """
        Returns a string representation of the MMMData object.

        Returns:
            str: A formatted string containing key information about the MMMData object.
        """
        data_info = (
            f"MMMData:\n"
            f"DataFrame Info:\n"
            f"  Shape: {self.data.shape}\n"
            f"  Columns: {', '.join(self.data.columns)}\n"
            f"  Date Range: {self.data[self.mmmdata_spec.date_var].min()} to {self.data[self.mmmdata_spec.date_var].max()}\n"
            f"\nDependent Variable:\n"
            f"  Name: {self.mmmdata_spec.dep_var}\n"
            f"  Type: {self.mmmdata_spec.dep_var_type}\n"
            f"  Summary Stats:\n"
            f"    Mean: {self.data[self.mmmdata_spec.dep_var].mean():.2f}\n"
            f"    Min: {self.data[self.mmmdata_spec.dep_var].min():.2f}\n"
            f"    Max: {self.data[self.mmmdata_spec.dep_var].max():.2f}\n"
            f"\nMedia Variables:\n"
            f"  Paid Media Spends: {self.mmmdata_spec.paid_media_spends}\n"
            f"  Paid Media Variables: {self.mmmdata_spec.paid_media_vars}\n"
            f"  Organic Variables: {self.mmmdata_spec.organic_vars}\n"
            f"\nOther Variables:\n"
            f"  Context Variables: {self.mmmdata_spec.context_vars}\n"
            f"  Factor Variables: {self.mmmdata_spec.factor_vars}\n"
            f"\nTime Window Info:\n"
            f"  Window Start: {self.mmmdata_spec.window_start}\n"
            f"  Window End: {self.mmmdata_spec.window_end}\n"
            f"  Interval Type: {self.mmmdata_spec.interval_type}\n"
            f"  Day Interval: {self.mmmdata_spec.day_interval}"
        )
        return data_info

    def display_data(self) -> None:
        """
        Display the contents of the DataFrame.
        """
        print(self.data)

    def get_summary(self) -> pd.DataFrame:
        """
        Get a summary of the DataFrame, including basic statistics.

        :return: A pandas DataFrame containing the summary statistics.
        """
        return self.data.describe()

    def add_column(self, column_name: str, data: List[Any]) -> None:
        """
        Add a new column to the DataFrame.

        :param column_name: The name of the new column.
        :param data: The data for the new column.
        """
        self.data[column_name] = data

    def remove_column(self, column_name: str) -> None:
        """
        Remove a column from the DataFrame.

        :param column_name: The name of the column to remove.
        """
        self.data.drop(columns=[column_name], inplace=True)

    def calculate_rolling_window_indices(self) -> None:
        """Match R's window index calculation exactly"""
        # Convert dates to pandas datetime
        self.data[self.mmmdata_spec.date_var] = pd.to_datetime(
            self.data[self.mmmdata_spec.date_var]
        )
        dates_vec = self.data[self.mmmdata_spec.date_var]

        # Convert window dates to datetime
        window_start = pd.to_datetime(self.mmmdata_spec.window_start)
        window_end = pd.to_datetime(self.mmmdata_spec.window_end)

        # Calculate start index using days difference like R
        if window_start is not None:
            days_diff = (dates_vec - window_start).dt.total_seconds() / (24 * 3600)
            closest_start_idx = days_diff.abs().argmin()
            closest_start_date = dates_vec.iloc[closest_start_idx]

            print(f"Python window calculation (start):")
            print(f"window_start: {window_start}")
            print(f"closest_start_date: {closest_start_date}")
            print(f"closest_start_idx: {closest_start_idx}")
            print(f"days_diff first few values: {days_diff.head().values}")

            self.mmmdata_spec.rolling_window_start_which = closest_start_idx
            self.mmmdata_spec.window_start = closest_start_date

        # Calculate end index using days difference like R
        if window_end is not None:
            days_diff = (dates_vec - window_end).dt.total_seconds() / (24 * 3600)
            closest_end_idx = days_diff.abs().argmin()
            closest_end_date = dates_vec.iloc[closest_end_idx]

            print(f"Python window calculation (end):")
            print(f"window_end: {window_end}")
            print(f"closest_end_date: {closest_end_date}")
            print(f"closest_end_idx: {closest_end_idx}")
            print(f"days_diff first few values: {days_diff.head().values}")

            self.mmmdata_spec.rolling_window_end_which = closest_end_idx
            self.mmmdata_spec.window_end = closest_end_date

        # Calculate window length
        if window_start is not None and window_end is not None:
            self.mmmdata_spec.rolling_window_length = (
                self.mmmdata_spec.rolling_window_end_which
                - self.mmmdata_spec.rolling_window_start_which
                + 1
            )
            print(f"Window length: {self.mmmdata_spec.rolling_window_length}")

    def set_default_factor_vars(self) -> None:
        """
        Set the default factor variables.
        """
        factor_variables = self.mmmdata_spec.factor_vars
        selected_columns = self.data[self.mmmdata_spec.context_vars]
        non_numeric_columns = ~selected_columns.applymap(
            lambda x: isinstance(x, (int, float))
        ).all()
        if non_numeric_columns.any():
            non_factor_columns = non_numeric_columns[
                ~non_numeric_columns.index.isin(factor_variables or [])
            ]
            non_factor_columns = non_factor_columns[non_factor_columns]
            if len(non_factor_columns) > 0:
                factor_variables = (
                    factor_variables or []
                ) + non_factor_columns.index.tolist()
        self.mmmdata_spec.factor_vars = factor_variables
