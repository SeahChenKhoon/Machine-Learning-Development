import pandas as pd
import numpy as np
import yaml
from datetime import datetime
from typing import Any, Dict

def read_yaml(yaml_filepath:str) -> Dict[str, Any]:
    """
    Reads a YAML file and returns the loaded data.

    Parameters:
        yaml_filepath (str): The path to the YAML file.

    Returns:
        Dict[str, Any]: A dictionary representing the loaded YAML data.

    Example:
        yaml_data = util.read_yaml("../config.yaml")
    """
    #Open yaml file
    with open(yaml_filepath, 'r') as file:
        # return configurable data from yaml file
        return yaml.safe_load(file)

def util_rm_col(df_dataframe: pd.DataFrame, list_cols:list[str])->None:
    """
    Remove specified columns from a pandas DataFrame in-place.

    Parameters:
        df_dataframe (pd.DataFrame): The DataFrame from which columns will be removed.
        list_cols (List[str]): A list of column names to be removed.

    Returns:
        None
    """
    df_dataframe.drop(list_cols, axis=1,inplace=True)
    return None


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
        
