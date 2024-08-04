# pyre-strict

from typing import List, Optional, Any
import pandas as pd

class MMMData:
    class MMMDataSpec:
        def __init__(
            self,
            dep_var: Optional[str] = None,
            dep_var_type: Optional[str] = None,
            date_var: str = "auto",
            paid_media_spends: Optional[List[str]] = None,
            paid_media_vars: Optional[List[str]] = None,
            paid_media_signs: Optional[List[str]] = None,
            organic_vars: Optional[List[str]] = None,
            organic_signs: Optional[List[str]] = None,
            context_vars: Optional[List[str]] = None,
            context_signs: Optional[List[str]] = None,
            factor_vars: Optional[List[str]] = None,
            adstock: Optional[str] = None
        ) -> None:
            self.dep_var: Optional[str] = dep_var
            self.dep_var_type: Optional[str] = dep_var_type
            self.date_var: str = date_var
            self.paid_media_spends: Optional[List[str]] = paid_media_spends
            self.paid_media_vars: Optional[List[str]] = paid_media_vars
            self.paid_media_signs: Optional[List[str]] = paid_media_signs
            self.organic_vars: Optional[List[str]] = organic_vars
            self.organic_signs: Optional[List[str]] = organic_signs
            self.context_vars: Optional[List[str]] = context_vars
            self.context_signs: Optional[List[str]] = context_signs
            self.factor_vars: Optional[List[str]] = factor_vars
            self.adstock: Optional[str] = adstock

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
            adstock: {self.adstock}
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
        :param mmmdata_spec: An MMMDataSpec object.
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

# Example usage:
if __name__ == "__main__":
    # Create a sample DataFrame
    sample_data: dict[str, List[int]] = {
        'A': [1, 2, 3],
        'B': [4, 5, 6],
        'C': [7, 8, 9]
    }
    df: pd.DataFrame = pd.DataFrame(sample_data)

    # Initialize MMMDataSpec
    mmmdata_spec: MMMData.MMMDataSpec = MMMData.MMMDataSpec(
        dep_var="sales",
        dep_var_type="revenue",
        paid_media_spends=["tv", "radio", "print"],
        paid_media_vars=["tv_GRP", "radio_GRP", "print_GRP"],
        prophet_country="US"
    )

    # Initialize MMMData with the sample DataFrame and MMMDataSpec
    mmm_data: MMMData = MMMData(df, mmmdata_spec)

    # Display the data
    mmm_data.display_data()

    # Get summary statistics
    summary: pd.DataFrame = mmm_data.get_summary()
    print(summary)

    # Add a new column
    mmm_data.add_column('D', [10, 11, 12])
    mmm_data.display_data()

    # Remove a column
    mmm_data.remove_column('A')
    mmm_data.display_data()

    # Print MMMDataSpec
    print(mmm_data.mmmdata_spec)

    # Update some attributes in MMMDataSpec
    mmm_data.mmmdata_spec.update(dep_var="conversions", dep_var_type="count", prophet_country="UK")

    # Print updated MMMDataSpec
    print(mmm_data.mmmdata_spec)
