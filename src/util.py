import pandas as pd
import numpy as np
import yaml
from datetime import datetime
from typing import Any

def timer(start_time=datetime)->datetime:
    """
    Measure and print the elapsed time.

    This function calculates and prints the elapsed time since the provided `start_time`.

    Parameters:
        start_time (datetime, optional): The starting time to calculate the elapsed time from.
            If not provided, the current time will be used as the starting time.

    Returns:
        datetime: The `start_time` when provided, or the current time if `start_time` is None.\
    """
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print("\n Time taken: %i hours %i minutes and %s seconds." % (thour, tmin, round(tsec,2)))
        return

def output_csv (data_path:str,dataframe:pd.DataFrame,dateframe_name:str)->None:
    """
    This function output a csv from a dataframe and store in data folder

    Parameters:
        col_name (str): Output filename of the csv
        dataframe (pd.DataFrame): Dataset in which to output to csv

    Returns:
        dataframe (pd.DataFrame): Return back the processed dataset    
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M")
    filename = f"{data_path}{dateframe_name}_{timestamp}.csv"
    dataframe.to_csv(filename)
    return

def drop_columns(dataframe:pd.DataFrame, col_names:list)->pd.DataFrame:
    """
    Drop specified columns from a pandas DataFrame.

    Parameters:
    - dataframe (pd.DataFrame): The DataFrame from which columns will be dropped.
    - col_names (list of str): A list of column names to be dropped from the DataFrame.

    Returns:
    - pd.DataFrame: A new DataFrame with the specified columns removed.
    """
    for col_name in col_names:
        dataframe.drop(col_name, axis=1,inplace=True)
    return dataframe

def read_config_file(config_file_path: str)-> Any:
    """
    Read and load configuration data from a YAML file.

    This function reads and loads configuration data from a YAML file located at the specified
    `config_file_path`.

    Parameters:
        config_file_path (str): The path to the YAML configuration file.

    Returns:
        Any: The loaded configuration data. The data type may vary depending on the content of
        the YAML file.
    """
    with open(config_file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

