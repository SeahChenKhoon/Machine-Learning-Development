import pandas as pd
import util
import numpy as np
import ast
from sklearn.preprocessing import LabelEncoder

class DataProcessing:
    def impute_missing_value_info (self, df: pd.DataFrame, impute_missing_val_info: list) -> pd.DataFrame:
        """
        Impute missing values in the DataFrame based on the provided information.

        Parameters:
        - df (pd.DataFrame): The DataFrame with missing values to be imputed.
        - impute_missing_val_info (list): A list containing dictionaries with information for imputing missing values.
            Each dictionary should have:
            - 'col_list' (list): A list of column names to impute missing values.
            - 'impute_type' (str): The imputation type ('mean', 'median', 'mode', etc.).

        Returns:
        - pd.DataFrame: The DataFrame with missing values imputed based on the provided information.

        Example:
        - dp.impute_missing_value_info(df_cruise,[{'impute_type': 'random', 'col_list': "['Gender','Online Check-in']"}, {'impute_type': 'mode', 'col_list': "['Cruise Name','Onboard Wifi Service']"}, {'impute_type': 'mean', 'col_list': "['Distance']"}])

        """
        for impute_missing_val in impute_missing_val_info:
            # Impute missing values for the specified columns using the provided imputation type
            df = self.impute_missing_value(df, ast.literal_eval(impute_missing_val['col_list']), impute_missing_val['impute_type'])
        return df

    def impute_missing_value(self, df: pd.DataFrame, col_list: list, impute_type: str) -> pd.DataFrame:
        """
        Impute missing values in specific columns of the DataFrame using the specified imputation strategy.

        Parameters:
        - df (pd.DataFrame): The DataFrame with missing values to be imputed.
        - col_list (list): A list of column names to impute missing values.
        - impute_type (str): The imputation strategy ('random', 'mode', 'mean', etc.).

        Returns:
        - pd.DataFrame: The DataFrame with missing values imputed in the specified columns.

        Example:
        - dp.impute_missing_value(df_cruise, ['Gender','Online Check-in'], "random")
        """
        if impute_type == "random":
            # Set a seed for reproducibility
            np.random.seed(42)
            for col_name in col_list:
                # For each column, fill missing values with a random choice from unique non-null values
                unique_values = df[col_name].dropna().unique()
                df[col_name] = df[col_name].apply(lambda x: np.random.choice(unique_values) if pd.isnull(x) else x)
        elif impute_type=="mode":
            # For each column, fill missing values with the mode (most frequent value)
            for col_name in col_list:
                mode_gender = df[col_name].mode().iloc[0]
                df[col_name].fillna(mode_gender, inplace=True)
        elif impute_type=="mean":
            # For each column, fill missing values with the mean
            for col_name in col_list:
                df[col_name].fillna(df[col_name].mean(), inplace=True)
        return df

    def label_encoder(self, df: pd.DataFrame, list_cols: list) -> pd.DataFrame:
        """
        Apply label encoding to specified columns in the DataFrame.

        Parameters:
        - df (pd.DataFrame): The DataFrame to be processed.
        - list_cols (list): A list of column names to be label encoded.

        Returns:
        - pd.DataFrame: The DataFrame with label encoding applied to the specified columns.

        Example:
        - dp.label_encoder(df_cruise, ['Gender','Onboard Wifi Service'])
        """
        label_encoder = LabelEncoder()
        
        # Iterate over the list of columns to apply label encoding
        for col in ast.literal_eval(list_cols):
            # Convert column values to strings and apply label encoding
            df[col] = label_encoder.fit_transform(df[col].astype(str))
        return df
    
    def numeric_conversion(self, df: pd.DataFrame, numeric_field_info: list) -> pd.DataFrame:
        """
        Convert specified columns in the DataFrame to numeric types.

        Parameters:
        - df (pd.DataFrame): The DataFrame to be processed.
        - numeric_field_info (list): A list of dictionaries, each containing information for numeric conversion.
            Each dictionary should have:
                - 'col_list' (list): A list of column names to be converted.
                - 'dtype' (str): The target numeric data type ('int32', 'float64', etc.).

        Returns:
        - pd.DataFrame: The DataFrame with specified columns converted to numeric types.

        Example:
        - dp.numeric_conversion(df_cruise, [{'col_list': "['Distance']", 'dtype': 'float64'}])
        """
        for numeric_field_info in numeric_field_info:
            # Call the convert_number method for each set of columns and data type
            self.convert_number(df, ast.literal_eval(numeric_field_info['col_list']), 
                                 numeric_field_info['dtype'])
        return df

    def convert_number(self, df: pd.DataFrame, col_list_info: list, dtype: str) -> pd.DataFrame:
        """
        Convert specified columns in the DataFrame to the specified numeric data type.

        Parameters:
        - df (pd.DataFrame): The DataFrame to be processed.
        - col_list_info (list): A list of column names to be converted.
        - dtype (str): The target numeric data type ('int32', 'float64', etc.).

        Returns:
        - pd.DataFrame: The DataFrame with specified columns converted to the specified numeric data type.

        Example:
        - dp.convert_number(df_cruise,['Distance'], 'float64')
        """
        for col_name in col_list_info:
            # Convert the specified column to the specified numeric data type
            if dtype == 'int32':
                df[col_name] = df[col_name].astype('Int32')
            elif dtype == 'float64':
                df[col_name] = df[col_name].astype('Float64')
        return df


    def valid_data_processing(self, df: pd.DataFrame, valid_data_info: list) -> pd.DataFrame:
        """
        Process valid data in specified columns of the DataFrame.

        Parameters:
        - df (pd.DataFrame): The DataFrame to be processed.
        - valid_data_info (list): A list of dictionaries, each containing information for valid data processing.
            Each dictionary should have:
                - 'col_list' (list): A list of column names to be restricted.
                - 'valid_data_list' (list): A list of valid data values.

        Returns:
        - pd.DataFrame: The DataFrame with specified columns processed for valid data.
        
        Example:
        - dp.valid_data_processing(df_cruise, 
            [{'col_list': "['Gender']", 'valid_data_list': "[None, 'Female','Male']"}])
        """
        for valid_data_col in valid_data_info:
            # Call the restrict_val method for each set of columns and valid data values
            df = self.restrict_val(df, ast.literal_eval(valid_data_col['col_list']), 
                                 ast.literal_eval(valid_data_col['valid_data_list']))
        return df

    def restrict_val(self, df: pd.DataFrame, col_list: list, valid_data_list: list) -> pd.DataFrame:
        """
        Restrict values in specified columns of the DataFrame to a predefined list of valid data.

        Parameters:
        - df (pd.DataFrame): The DataFrame to be processed.
        - col_list (list): A list of column names to be restricted.
        - valid_data_list (list): A list of valid data values.

        Returns:
        - pd.DataFrame: The DataFrame with specified columns restricted to valid data values.

        Example:
        - dp.restrict_val(df, ['Gender'], [None, 'Female','Male'])
        """
        for col_name in col_list:
            # Restrict values in the specified column to the valid data list
            col_dtype = df[col_name].dtype
            df = df[df[col_name].isin(valid_data_list)]
        return df

    def split_composite_field(self, dataframe: pd.DataFrame, composite_fields: list) -> pd.DataFrame:
        """
        Split composite fields in the DataFrame into individual columns.

        Parameters:
        - dataframe (pd.DataFrame): The DataFrame to be processed.
        - composite_fields (list): A list of dictionaries, each containing information for splitting composite fields.
            Each dictionary should have:
                - 'composite_field' (str): The name of the composite field to be split.
                - 'new_column_list' (list): A list of new column names to be created from the split values.
                - 'delimiter' (str): The delimiter used to split the composite field.

        Returns:
        - pd.DataFrame: The DataFrame with composite fields split into individual columns.

        Example:
        - dp.split_composite_field(df_cruise,[{'composite_field': 'Source of Traffic', 'new_column_list': "['Source', 'Traffic']", 'delimiter': ' - '}])
        """
        if composite_fields:
            for composite_field in composite_fields:
                self.split_col(dataframe,composite_field['composite_field'], ast.literal_eval(composite_field['new_column_list']), 
                            composite_field['delimiter'])
        return dataframe

    def split_col(self, dataframe: pd.DataFrame, composite_col: str, list_cols: list, delimiter: str) -> pd.DataFrame:
        """
        Split a composite column into a list of values and create new columns from the list.

        Parameters:
        - dataframe (pd.DataFrame): The DataFrame to be processed.
        - composite_col (str): The name of the composite column to be split.
        - list_cols (list): A list of new column names to be created from the split values.
        - delimiter (str): The delimiter used to split the composite column.

        Returns:
        - pd.DataFrame: The DataFrame with new columns created from the split values, 
        and the original composite column dropped.
        
        Example:
        - dp.split_col(df_cruise,['Source of Traffic', ['Source', 'Traffic'], ' - ')
        """
        # Split the composite column into a list of values
        split_values = dataframe[composite_col].str.split(delimiter)

        # Create new columns from the list of values
        for i, new_col in enumerate(list_cols):
            dataframe[new_col] = split_values.str[i]

        # Drop the original composite column
        dataframe.drop(columns=[composite_col], inplace=True)
        return dataframe
    
    def dirty_data_processing(self, df: pd.DataFrame, dirty_data_info: list) -> pd.DataFrame:
        """
        Process dirty data in specified columns of the DataFrame.

        Parameters:
        - df (pd.DataFrame): The DataFrame to be processed.
        - dirty_data_info (list): A list of dictionaries, each containing information for processing dirty data.
            Each dictionary should have:
                - 'field_list' (list): A list of column names to be processed.
                - 'replace_val' (any): The value to be replaced in the specified columns.
                - 'replace_with' (any): The value to replace with in the specified columns.
                - 'like_ind' (bool): A flag indicating whether to perform a substring match.

        Returns:
        - pd.DataFrame: The DataFrame with specified columns processed for dirty data.

        Example:
        - dp.dirty_data_processing(df_cruise,[{'field_list': "['Gate location']", 'replace_val': 0, 'replace_with': 'None', 'like_ind': False}])
        """
        for dirty_data in dirty_data_info:
            # Call the replace_value method for each set of columns and dirty data information
            self.replace_value(df, ast.literal_eval(dirty_data['field_list']), dirty_data['replace_val'],  
                                   dirty_data['replace_with'], dirty_data['like_ind'])
        return df
    
    def replace_value(self, df: pd.DataFrame, col_list: list[str], replace_val: any, replace_with: any,
                      like_ind: bool = False) -> pd.DataFrame:
        """
        Replace values in specified columns of the DataFrame based on the provided conditions.

        Parameters:
        - df (pd.DataFrame): The DataFrame to be processed.
        - col_list (list): A list of column names to be processed.
        - replace_val (any): The value to be replaced in the specified columns.
        - replace_with (any): The value to replace with in the specified columns.
        - like_ind (bool): A flag indicating whether to perform a substring match.

        Returns:
        - pd.DataFrame: The DataFrame with specified columns processed for value replacement.

        Example:
        - dp.replace_value(df,['Gate location'], 0, None, False)
        """
        for col_name in col_list:
            # Check if substring match is required
            if like_ind == False:
                # If replace_with is "None", set it to None
                if replace_with == "None":
                    replace_with = None
                # Replace values in the specified column with the provided replace_with value
                df.loc[df[col_name]==replace_val,col_name] = replace_with
            else:
                # Calculate the length of the substring to match
                str_len = len(replace_val)
                # Create a temporary column 'substring' containing the first 'str_len' characters of the original column
                df['substring'] = df[col_name].str[:str_len]
                # Replace values in the specified column where the uppercase substring matches the uppercase replace_val
                df.loc[df['substring'].str.upper()== replace_val.upper(), col_name] = replace_with
                # Remove the temporary 'substring' column
                util.util_rm_col(df,'substring')
        return df 

    def replace_nan_none(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Replace NaN values in the DataFrame with None.

        Parameters:
        - df (pd.DataFrame): The DataFrame to be processed.

        Returns:
        - pd.DataFrame: The DataFrame with NaN values replaced by None.

        Example:
        - dp.replace_nan_none(df_cruise)
        """
        # Use the replace method to replace NaN with None
        df.replace({np.nan: None},inplace=True)
        return df

    def rm_cols_high_missing(self, df: pd.DataFrame, threshold: float) -> pd.DataFrame:
        """
        Remove columns with missing values exceeding the specified threshold.

        Parameters:
        - df (pd.DataFrame): The DataFrame to be processed.
        - threshold (float): The threshold percentage for missing values.

        Returns:
        - pd.DataFrame: The DataFrame with high-missing columns removed.

        Example:
        - dp.rm_cols_high_missing(df_cruise, 0.4)
        """
        # Calculate the percentage of missing values for each column
        missing_percentages = df.isnull().mean()
        # Identify columns exceeding the threshold
        columns_to_remove = missing_percentages[missing_percentages > threshold].index
        # Drop columns with high missing values
        df.drop(columns=columns_to_remove, inplace=True)
        return df

    def rm_rows_target_var(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """
        Remove rows with missing values in the specified target column.

        Parameters:
        - df (pd.DataFrame): The DataFrame to be processed.
        - target_col (str): The name of the target column.

        Returns:
        - pd.DataFrame: The DataFrame with rows containing missing values in the target column removed.

        Example:
        - dp.rm_rows_target_var(df_cruise, "Ticket Type")
        """
        # Remove rows with missing values in target columns
        df.dropna(subset=target_col, inplace=True)
        return df

    def remove_missing(self, df: pd.DataFrame, list_cols: list) -> pd.DataFrame:
        """
        Remove rows with missing values in the specified columns.

        Parameters:
        - df (pd.DataFrame): The DataFrame to be processed.
        - list_cols (list): A list of column names to be considered for missing value removal.

        Returns:
        - pd.DataFrame: The DataFrame with rows containing missing values in the specified columns removed.

        Example:
        - dp.remove_missing(df_cruise, ['Ticket Type'])
        """
        # Remove rows with missing values in specified columns
        df.dropna(subset=list_cols, inplace=True)
        return df

    def obj_to_datetime(self, df: pd.DataFrame, datetime_fields_info: list) -> pd.DataFrame:
        """
        Convert specified columns to datetime format based on the provided information.

        Parameters:
        - df (pd.DataFrame): The DataFrame to be processed.
        - datetime_fields_info (list): A list of dictionaries, each containing information for datetime conversion.
            Each dictionary should have:
                - 'column_list' (list): A list of column names to be converted.
                - 'format' (str): The datetime format string.

        Returns:
        - pd.DataFrame: The DataFrame with specified columns converted to datetime format.

        Example:
        - dp.obj_to_datetime(df_cruise, ['Date_of_Logging'])
        """
        if datetime_fields_info:
            # Iterate through each dictionary in datetime_fields_info
            for datetime_field_info in datetime_fields_info:
                # Extract column names and format from the dictionary
                col_names = ast.literal_eval(datetime_field_info['column_list'])
                # Iterate through each column name and convert it to datetime format
                for col_name in col_names:
                    # Convert the specified column to datetime format
                    df[col_name] = pd.to_datetime(df[col_name], format=datetime_field_info['format'], errors='coerce')
        return df

    def rm_id_cols(self, df, list_cols: list[str]) -> pd.DataFrame:
        """
        Remove specified columns from the DataFrame.

        Parameters:
        - df (pd.DataFrame): The DataFrame to be processed.
        - list_cols (list): A list of column names to be removed.

        Returns:
        - pd.DataFrame: The DataFrame with specified columns removed.

        Example:
        - dp.rm_id_cols(df_cruise, ['Ext_Intcode'])
        """
        # Use the utility function to remove specified columns
        util.util_rm_col(df, list_cols)
        return df
    


