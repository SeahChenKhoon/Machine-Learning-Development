import pandas as pd
import numpy as np
import seaborn as sns
import ast
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import util

class FeatureEngineering:
    def yyyy_from_date(self, df: pd.DataFrame, date_yyyy_info: list) -> pd.DataFrame:
        """
        Extract the year from specified date columns and store the result in new columns.

        Parameters:
        - df (pd.DataFrame): The DataFrame to be processed.
        - date_yyyy_info (list): A list of dictionaries, each containing information for date-to-year conversion.
            Each dictionary should have:
                - 'col_list' (list): A list of column names containing date values.
                - 'yyyy_col_list' (list): A list of new column names to store the extracted years.

        Returns:
        - pd.DataFrame: The DataFrame with new columns containing the extracted years.

        Example:
        - fe.yyyy_from_date(df_cruise,{'col_list': "['Date of Birth','Logging']", 'yyyy_col_list': "['Year of Birth','Year of Logging']"})
        """
        # Call the convert_datetime_to_year method for each set of date and year columns
        self.convert_datetime_to_year(df, ast.literal_eval(date_yyyy_info['col_list']), ast.literal_eval(date_yyyy_info['yyyy_col_list']))
        return df

    def convert_datetime_to_year(self, df: pd.DataFrame, date_cols: list, yyyy_cols: list) -> None:
        """
        Extract the year from specified date columns and store the result in new columns.

        Parameters:
        - df (pd.DataFrame): The DataFrame to be processed.
        - date_cols (list): A list of column names containing date values.
        - yyyy_cols (list): A list of new column names to store the extracted years.

        Returns:
        - None

        Example:
        - fe.convert_datetime_to_year(df_cruise, ['Date of Birth','Logging'], ['Year of Birth','Year of Logging'])
        """
        for date_col, yyyy_col in zip(date_cols, yyyy_cols):
            # Extract the year from the date column and store it in the corresponding year column
            df[yyyy_col] = df[date_col].dt.year

    def one_hot_key_encode(self, df: pd.DataFrame, col_names: list[str]) -> pd.DataFrame:
        """
        Perform one-hot encoding on specified columns.

        Parameters:
        - df (pd.DataFrame): The DataFrame to be processed.
        - col_names (list): A list of column names to be one-hot encoded.

        Returns:
        - pd.DataFrame: The DataFrame with one-hot encoded columns.

        Example:
        - fe.one_hot_key_encode(df_cruise, ['Traffic'])
        """
        # Use pd.get_dummies to perform one-hot encoding
        df = pd.get_dummies(df, columns=col_names, drop_first=True)
        return df

    def convert_miles_to_KM(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert distances from miles to kilometers for rows with 'Dist_Metrics' value equal to 1.

        Parameters:
        - df (pd.DataFrame): The DataFrame to be processed.

        Returns:
        - pd.DataFrame: The DataFrame with distances converted to kilometers.

        Example:
        - fe.convert_miles_to_KM(df_cruise)
        """
        # Use loc to identify rows where 'Dist_Metrics' is 1 and perform the conversion
        df.loc[df['Dist_Metrics'] == 1, 'Distance'] = df.loc[df['Dist_Metrics'] == 1, 'Distance'] * 1.609344
        return df

    def calc_year_diff(self, df, diff_years) -> pd.DataFrame:
        """
        Calculate the difference in years between specified columns and store the result in new columns.
        Remove the original columns after calculation.

        Parameters:
        - df (pd.DataFrame): The DataFrame to be processed.
        - diff_years (list): A list of dictionaries, each containing information for year difference calculation.
            Each dictionary should have:
                - 'new_col' (str): The name of the new column to store the year difference.
                - 'minuend_col' (str): The column representing the minuend (subtracted from).
                - 'subtrahend_col' (str): The column representing the subtrahend (subtracted).

        Returns:
        - pd.DataFrame: The DataFrame with new columns containing the year differences.

        Example:
        - fe.calc_year_diff(df_cruise, [{'minuend_col': 'Year of Logging', 'subtrahend_col': 'Year of Birth', 'new_col': 'Age'}])
        """
        # Iterate through each dictionary in diff_years
        for diff_year in diff_years:
            # Calculate the year difference and store it in the new column
            df[diff_year['new_col']] = df[diff_year['minuend_col']] - df[diff_year['subtrahend_col']]
            # Use the utility function to remove the original columns
            util.util_rm_col(df, [diff_year['minuend_col'], diff_year['subtrahend_col']])
        return df

    def denote_missing_col(self) -> None:
        """
        Create binary columns to indicate missing values in original columns.
        Create a total count column for missing values and remove intermediate missing value columns.

        Returns:
        - None

        Example:
        - denote_missing_col()
        """
        missing_col_list = []
        # Iterate through each column in the DataFrame
        for col_name in self.dataframe.columns:
            # Check if the column has missing values
            if self.dataframe[col_name].isnull().sum() > 0:
                # Create a new binary column indicating missing values
                self.dataframe[col_name + "_missing"] = self.dataframe[col_name].isnull().astype(int)
                missing_col_list.append(col_name + "_missing")

        # Create a new column "tot_missing_col" that represents the total number of missing values
        self.dataframe['tot_missing_col'] = self.dataframe[missing_col_list].sum(axis=1)
        # Use the utility function to remove the intermediate missing value columns
        util.util_rm_col(self.dataframe, missing_col_list)
        return None
            