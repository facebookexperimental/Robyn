# pyre-strict

from dataclasses import dataclass
from typing import List, Optional, Any
import pandas as pd
from robyn.data.entities.enums import ContextSigns, DependentVarType, OrganicSigns, PaidMediaSigns

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
            # window_start: datetime,
            # window_end: datetime,
            paid_media_spends: Optional[List[str]] = None,
            paid_media_vars: Optional[List[str]] = None,
            paid_media_signs: Optional[List[PaidMediaSigns]] = None,
            organic_vars: Optional[List[str]] = None,
            organic_signs: Optional[List[OrganicSigns]] = None,
            context_vars: Optional[List[str]] = None,
            context_signs: Optional[List[ContextSigns]] = None,
            factor_vars: Optional[List[str]] = None,
        ) -> None:
            self.dep_var: Optional[str] = dep_var
            self.dep_var_type: DependentVarType = dep_var_type
            self.date_var: str = date_var
            self.paid_media_spends: Optional[List[str]] = paid_media_spends
            self.paid_media_vars: Optional[List[str]] = paid_media_vars
            self.paid_media_signs: Optional[List[str]] = paid_media_signs
            self.organic_vars: Optional[List[str]] = organic_vars
            self.organic_signs: Optional[List[str]] = organic_signs
            self.context_vars: Optional[List[str]] = context_vars
            self.context_signs: Optional[List[str]] = context_signs
            self.factor_vars: Optional[List[str]] = factor_vars

        def __str__(self) -> str:
            return f"""
            MMMDataSpec:
            dep_var: {self.dep_var}
            dep_var_type: {self.dep_var_type}
            date_var: {self.date_var}
            paid_media_spends: {self.paid_media_spends}
            paid_media_vars: {self.paid_media_vars}
            paid_media_signs: {self.paid_media_signs}
            organic_vars: {self.organic_vars}
            organic_signs: {self.organic_signs}
            context_vars: {self.context_vars}
            context_signs: {self.context_signs}
            factor_vars: {self.factor_vars}
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
                    raise AttributeError(f"{key} is not a valid attribute of MMMDataSpec")

    def __init__(self, data: pd.DataFrame, mmmdata_spec: MMMDataSpec) -> None:
        """
        Initialize the MMMData class with a pandas DataFrame and an MMMDataSpec object.

        :param data: A pandas DataFrame containing the data.
        :param mmmdata_spec: An MMMDataSpec object containing mapping of what is what in the provided data.
        """
        self.data: pd.DataFrame = data
        self.mmmdata_spec: MMMData.MMMDataSpec = mmmdata_spec

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
