import pandas as pd
import numpy as np
import sqlite3
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

# def convert_object_to_datetime(dataframe: pd.DataFrame, col_name:str)-> pd.DataFrame:
#     """
#         This function process the Date of Birth by
#             a> Remove invalid Datatime value in column
#             b> Convert column to datetime format

#     Parameters:
#         col_name (str) and dataframe (pd.DataFrame): Specify the column name within dataframe for the function perform processing.
        

#     Returns:
#         dataframe (pd.DataFrame): Return back the processed dataset
#     """
#     remove_non_dd_mm_yyyy_format(dataframe, col_name)
#     convert_column_to_datetime(dataframe,col_name)
#     return dataframe

# def fix_typo_error(dataframe:pd.DataFrame, col_name:str, replace_list:list, replace_with:str) -> pd.DataFrame:
#     """
#     Perform fixing of typo error list of data (replace_list) to replace_with 

#     Parameters:
#         col_name (str) and dataframe (pd.DataFrame): Specify the column name within dataframe for the function perform processing.
#         replace_list (list): List of element to be replace
#         replace_with (str): String to replace the element in replace_list

#     Returns:
#         dataframe (pd.DataFrame): Return back the processed dataset 
#     """
#     for word in replace_list:
#         dataframe.loc[dataframe[col_name]==word,col_name] = replace_with
#     return dataframe

# def remove_record_with_missing_data(dataframe:pd.DataFrame, col_name:str) -> pd.DataFrame:
#     """
#     This function remove all rows with missing data based on the specified dataframe and col_name

#     Parameters:
#         col_name (str) and dataframe (pd.DataFrame): Specify the column name within dataframe for the function perform processing.
        

#     Returns:
#         dataframe (pd.DataFrame): Return back the processed dataset
#     """
#     dataframe.dropna(subset=[col_name], inplace=True)
#     return dataframe

# def impute_mode(dataframe:pd.DataFrame, col_name)->pd.DataFrame:
#     """
#     Perform the following based on the specified column name and dataset 
#     1> the creation of new column col_name + "_mode" to track missing value indication
#     2> Impute the missing value with its mode

#     Parameters:
#         col_name (str) and dataframe (pd.DataFrame): Specify the column name within dataframe for the function perform processing.
        

#     Returns:
#         dataframe (pd.DataFrame): Return back the processed dataset
#     """
#     dataframe[col_name + "_mode"] = np.where(dataframe[col_name].isnull(),1,0)
#     dataframe[col_name] = dataframe[col_name].fillna(dataframe[col_name].mode()[0])
#     return dataframe

# def impute_random(dataframe, col_name)->pd.DataFrame:
#     """
#     Perform the following based on the specified column name and dataset 
#     1> the creation of new column col_name + "_random" to track missing value indication
#     2> Impute the missing value with its random value within the existing value range

#     Parameters:
#         col_name (str) and dataframe (pd.DataFrame): Specify the column name within dataframe for the function perform processing.
        

#     Returns:
#         dataframe (pd.DataFrame): Return back the processed dataset
#     """
#     dataframe[col_name + "_rand"] = np.where(dataframe[col_name].isnull(),1,0)
#     random_values = dataframe[col_name].dropna().sample(dataframe[col_name].isna().sum(), replace=True).values
#     dataframe.loc[dataframe[col_name].isna(), col_name] = random_values
#     return dataframe

# def impute_mean(dataframe:pd.DataFrame, col_name)->pd.DataFrame:
#     """
#     Perform the following based on the specified column name and dataset 
#     1> the creation of new column col_name + "_mean" to track missing value indication
#     2> Impute the missing value with its mean value 

#     Parameters:
#         col_name (str) and dataframe (pd.DataFrame): Specify the column name within dataframe for the function perform processing.
        

#     Returns:
#         dataframe (pd.DataFrame): Return back the processed dataset
#     """
#     dataframe[col_name + "_mean"] = np.where(dataframe[col_name].isnull(),1,0)
#     dataframe[col_name] = dataframe[col_name].fillna(int(dataframe[col_name].mean()))
#     return dataframe

# def impute_median(dataframe:pd.DataFrame, col_name)->pd.DataFrame:
#     dataframe[col_name + "_median"] = np.where(dataframe[col_name].isnull(),1,0)
#     dataframe[col_name] = dataframe[col_name].fillna(int(dataframe[col_name].median()))
#     return dataframe

# # ******************************

    
# def remove_duplicates_keeping_last (dataframe:pd.DataFrame, col_name:str,idx_col:str)->None:
#     """
#     Perform Removal duplicate records while retaining the last occurrence..

#     Parameters:
#         dataframe (pd.DataFrame): The DataFrame containing the column from which duplicates should be removed.
#         col_name (str): The name of the column within the DataFrame from which duplicates should be removed..

#     Returns:
#         dataframe (pd.DataFrame): Return back the processed dataset
#     """
#     dataframe.reset_index(inplace=True)
#     all_dup_idx = set(dataframe[idx_col].loc[dataframe.duplicated(subset=[col_name], keep=False)])
#     last_dup_idx = set(dataframe[idx_col].loc[dataframe.duplicated(subset=[col_name], keep="first")])
#     dup_idx_to_remove = all_dup_idx.symmetric_difference(last_dup_idx)
#     return dataframe


# def check_duplicates_in_column1(dataframe:pd.DataFrame, col_name:str)->bool:
#     """
#     Perform checking for duplicates with given dataframe and column nam
    
#     Parameters:
#         dataframe (pd.DataFrame): The DataFrame in which contains the column to check for duplicates.
            
#         col_name (str): The name of the column to be checked.

#     Returns:
#         duplicates (pd.DataFrame): duplicated dataset if duplicates found, None otherwise.
#     """
#     duplicates = dataframe[dataframe.duplicated(subset=[col_name], keep=False)]
    
#     if not duplicates.empty:
#         return duplicates
#     else:
#         return None

# def calculate_year_difference(dataframe:pd.DataFrame, col_name:str, new_col_name:str)->pd.DataFrame:
#     """
#     Perform computation of year using the formula current year minus dataframe[col_name]

#     Parameters:
#         dataframe (pd.DataFrame): The DataFrame in which the function execute on.
#         col_name (str): The name of the column which the function to execute on.
#         new_col_name (str): The new column which contain the year difference

#     Returns:
#         pd.DataFrame: The new DataFrame having the new computed year difference.
#     """
#     current_year = datetime.datetime.now().year
#     dataframe[new_col_name] = current_year - dataframe[col_name].dt.year
#     return dataframe

# def convert_column_to_datetime(df:pd.DataFrame, column_name:str, date_format:str='%d/%m/%Y')->pd.DataFrame:
#     """
#     Perform conversion of specified col_name in dataframe from Object to DateTime.

#     Parameters:
#         col_name (str) and dataframe (pd.DataFrame): Specify the column name within dataframe for the function perform processing.
#         date_format (str): Specify the date time format to convert

#     Returns:
#         dataframe (pd.DataFrame): Return back the processed dataset
#     """
#     # df[column_name] = pd.to_datetime(df[column_name], format=date_format)
#     return df

# def remove_non_dd_mm_yyyy_format(dataframe:pd.DataFrame, column_name:str)->pd.DataFrame:
#     """
#     This function removes all rows that have invalid datetime give the dataframe and column name

#     Parameters:
#         col_name (str) and dataframe (pd.DataFrame): Specify the column name within dataframe for the function perform processing.
        

#     Returns:
#         dataframe (pd.DataFrame): Return back the processed dataset
        
#     """
#     try:
#         dataframe[column_name] = pd.to_datetime(dataframe[column_name], format='%d/%m/%Y', errors='coerce')
#         dataframe = dataframe.dropna(subset=[column_name])
#     except ValueError:
#         print(f"Column '{column_name}' does not contain valid 'DD/MM/YYYY' dates.")

#     return dataframe

# def one_hot_encode(dataframe:pd.DataFrame, columns_to_encode:list)->pd.DataFrame:
#     """
#     Perform one-hot encoding on the specified columns of a DataFrame and concatenate the result.

#     Parameters:
#         df (pd.DataFrame): The DataFrame to be encoded.
#         columns_to_encode (list): A list of column names to be one-hot encoded.

#     Returns:
#         pd.DataFrame: The DataFrame with specified columns one-hot encoded and concatenated.
#     """
#     encoded_df = pd.get_dummies(dataframe, columns=columns_to_encode)
#     return encoded_df

# def map_values_to_binary(dataframe:pd.DataFrame, col_name:str, data_val:list)->pd.DataFrame:
#     """
#     Map specific values in a DataFrame column to 0 and 1.

#     Parameters:
#         df (pd.DataFrame): The DataFrame to be modified.
#         col (str): The column name to map values in.
#         data_val (list): A list of two values to map to 0 and 1, respectively.

#     Returns:
#         pd.DataFrame: The modified DataFrame.
#     """

#     if len(data_val) != 2:
#         print("data_val should contain exactly two values.")
#         return dataframe

#     # dataframe[col_name] = dataframe[col_name].map({data_val[0]: 0, data_val[1]: 1})
#     dataframe[col_name] = dataframe.apply(lambda row: 0 if row[col_name] == data_val[0] else (1 if row[col_name] == data_val[1] else row[col_name]), axis=1)
#     return dataframe

# def ordinal_encode(dataframe, column_name, categories_order):
#     """
#     Ordinally encodes a specified column in a DataFrame.

#     Parameters:
#         dataframe (pd.DataFrame): The DataFrame containing the column to encode.
#         column_name (str): The name of the column to encode.
#         categories_order (list): A list of categories in the desired order.

#     Returns:
#         pd.DataFrame: The DataFrame with the specified column ordinally encoded.
#     """
#     encoder = OrdinalEncoder(categories=[categories_order])
#     dataframe[column_name] = encoder.fit_transform(dataframe[[column_name]])
#     dataframe.loc[dataframe[column_name]==0,column_name] = None
#     return dataframe

# def label_encode_column(dataframe, col_name):
#     """
#     Perform label encoding on a specific column of a DataFrame.

#     Parameters:
#         dataframe (pd.DataFrame): The DataFrame to be encoded.
#         col_name (str): The name of the column to be label encoded.

#     Returns:
#         pd.DataFrame: The DataFrame with the specified column label encoded.
#     """
#     label_encoder = LabelEncoder()
#     dataframe[col_name] = label_encoder.fit_transform(dataframe[col_name])

#     return dataframe



