import pandas as pd
import numpy as np
import yaml
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
# import numpy as np
# from typing import Any

def read_yaml(yaml_filepath:str):
    # Read the YAML file
    with open(yaml_filepath, 'r') as file:
        data = yaml.safe_load(file)
        return data

def merge_dataframe(df_pre_cruise, df_post_cruise):
    return pd.merge(df_pre_cruise, df_post_cruise, left_index=True, right_index=True, how='inner')

def split_column(df_dataframe: pd.DataFrame, composite_col: str, list_cols: list, delimiter: str):
    # Split the composite column into a list of values
    split_values = df_dataframe[composite_col].str.split(delimiter)

    # Create new columns from the list of values
    for i, new_col in enumerate(list_cols):
        df_dataframe[new_col] = split_values.str[i]

    # Drop the original composite column
    df_dataframe.drop(columns=[composite_col], inplace=True)
    return None

def remove_missing_value(df_dataframe: pd.DataFrame, list_cols: list) -> None:
    # Remove rows with missing values in specified columns
    df_dataframe.dropna(subset=list_cols, inplace=True)

    return None

def label_encoder(df_dataframe: pd.DataFrame, list_cols: list) -> None:
    label_encoder = LabelEncoder()
    
    for col in list_cols:
        df_dataframe[col] = label_encoder.fit_transform(df_dataframe[col].astype(str))

    return None

def convert_object_to_datetime(df_dataframe: pd.DataFrame, list_cols:list[str], 
                             format:list[str])->None:
    count =0
    for col_name in list_cols:
        format_col = format[count]
        df_dataframe[col_name] = pd.to_datetime(df_dataframe[col_name], format=format_col, 
                                                   errors='coerce')
        count += 1
    return None

def convert_datetime_to_year(df_dataframe: pd.DataFrame, list_cols:list[str], list_new_cols:list)->None:
    count =0
    for col_name in list_cols:
        new_col = list_new_cols[count]
        df_dataframe[new_col] = df_dataframe[col_name].dt.year.astype(np.int32)
        count += 1
    remove_col(df_dataframe, list_cols)
    return None

def remove_col(df_dataframe: pd.DataFrame, list_cols:list[str])->None:
    df_dataframe.drop(list_cols, axis=1,inplace=True)

def print_type_value(data):
    print(type(data))
    print(data)

def output_csv (data_path:str,dataframe:pd.DataFrame,dateframe_name:str)->None:
    """
    This function output a csv from a dataframe and store in data folder

    Parameters:
        col_name (str): Output filename of the csv
        dataframe (pd.DataFrame): Dataset in which to output to csv

    Returns:
        dataframe (pd.DataFrame): Return back the processed dataset    
    """
    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    filename = f"{data_path}{dateframe_name}_{timestamp}.csv"
    dataframe.to_csv(filename)
    return
    
# def timer(start_time=datetime)->datetime:
#     """
#     Measure and print the elapsed time.

#     This function calculates and prints the elapsed time since the provided `start_time`.

#     Parameters:
#         start_time (datetime, optional): The starting time to calculate the elapsed time from.
#             If not provided, the current time will be used as the starting time.

#     Returns:
#         datetime: The `start_time` when provided, or the current time if `start_time` is None.\
#     """
#     if not start_time:
#         start_time = datetime.now()
#         return start_time
#     elif start_time:
#         thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
#         tmin, tsec = divmod(temp_sec, 60)
#         print("\n Time taken: %i hours %i minutes and %s seconds." % (thour, tmin, round(tsec,2)))
#         return



# def drop_columns(dataframe:pd.DataFrame, col_names:list)->pd.DataFrame:
#     """
#     Drop specified columns from a pandas DataFrame.

#     Parameters:
#     - dataframe (pd.DataFrame): The DataFrame from which columns will be dropped.
#     - col_names (list of str): A list of column names to be dropped from the DataFrame.

#     Returns:
#     - pd.DataFrame: A new DataFrame with the specified columns removed.
#     """
#     for col_name in col_names:
#         dataframe.drop(col_name, axis=1,inplace=True)
#     return dataframe

# def read_config_file(config_file_path: str)-> Any:
#     """
#     Read and load configuration data from a YAML file.

#     This function reads and loads configuration data from a YAML file located at the specified
#     `config_file_path`.

#     Parameters:
#         config_file_path (str): The path to the YAML configuration file.

#     Returns:
#         Any: The loaded configuration data. The data type may vary depending on the content of
#         the YAML file.
#     """
#     with open(config_file_path, 'r') as file:
#         config = yaml.safe_load(file)
#     return config

