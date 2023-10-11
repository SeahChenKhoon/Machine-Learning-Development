import pandas as pd
import numpy as np
import yaml
import datetime
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder



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

def read_config_file(config_file_path):
    with open(config_file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

