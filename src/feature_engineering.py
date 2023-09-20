import pandas as pd
import util
from datetime import datetime

# impt_order = [None, 'Not at all important', 'A little important', 'Somewhat important',
#     'Very important','Extremely important']

# def fe_convert_binary_columns(dataframe: pd.DataFrame, col_name:str, binary_val:list)-> pd.DataFrame:
#     """
#     Perform Binary encoding for the specific col_name in dataframe to set the first element to 0 and second to 1

#     Parameters:
#         dataframe (pd.DataFrame): The dataset in which the columns to be encoded
#         col_name (str): The columns to be encoded.
#         binary_val (list): A list contain 2 element to transform first element to 0 and second to 1
        
#     Returns:
#         dataframe (pd.DataFrame): Return back the processed dataset    
#     """
#     if len(binary_val) != 2:
#         print("This function expacts 2 elements in list.")
#         return dataframe
#     else:
#         dataframe = util.map_values_to_binary(dataframe,col_name,binary_val)
#     return dataframe

# def fe_age(dataframe: pd.DataFrame)-> pd.DataFrame:
#     """
#     Perform converstion of DOB to age by substract the year of birth with current year

#     Parameters:
#         col_name (str) and dataframe (pd.DataFrame): Specify the column name within dataframe for the function perform processing.
    
#     Returns:
#         dataframe (pd.DataFrame): Return back the processed dataset    
#     """
#     dataframe = util.calculate_year_difference(dataframe,"Date of Birth","Age")
#     return dataframe

# def fe_ordinal_encode(dataframe: pd.DataFrame, list_column:list)-> pd.DataFrame:
#     """
#     This function perform Ordinal Encode on the specified list of columns 
#         found in the specified dataframe
    
#     Parameters:
#         dataframe (pd.DataFrame): Specify dataset to be encode
#         list_column (list): Specify the list of importance
        
#     Returns:
#         dataframe (pd.DataFrame): Return back the processed dataset    
#     """
#     for col in list_column:
#         util.ordinal_encode(dataframe, col, impt_order)
#     return dataframe

# def fe_one_hot_key(dataframe: pd.DataFrame, col_name:list)-> pd.DataFrame:
#     """
#     This function process the one hot key encoding on the following column
#         a> Source of Traffic

#     Parameters:
#         col_name ( list) and dataframe (pd.DataFrame): Specify the column name within dataframe for the function perform processing.

#     Returns:
#         dataframe (pd.DataFrame): Return back the processed dataset    
#     """
#     dataframe =  util.one_hot_encode(dataframe,col_name)
#     return dataframe

# def fe_label_encode(dataframe: pd.DataFrame, col_name:str)-> None:
#     """
#     This function process the one hot key encoding on the following column
#             a> Ticket Type
    
#     Parameters:
#         col_name (str) and dataframe (pd.DataFrame): Specify the column name within dataframe for the function perform processing.
        
#     Returns:
#         dataframe (pd.DataFrame): Return back the processed dataset
#     """
#     dataframe =  util.label_encode_column(dataframe,col_name)
#     return dataframe

# def fe_impute_mode(dataframe:pd.DataFrame, col_names:list)->pd.DataFrame:
#     """
#     Interact with util.impute_mode to 
#     1> Replace the missing value with its mode
#     2> Create a new colume *_mode to indicate there was a missing value

#     Parameters:
#         col_name (list) and dataframe (pd.DataFrame): Specify the column names within dataframe for the function perform processing of mode.
        
#     Returns:
#         dataframe (pd.DataFrame): Return back the processed dataset    
#     """ 
#     for col_name in col_names:
#         dataframe = util.impute_mode(dataframe,col_name)
#     return dataframe

# def fe_impute_mean(dataframe:pd.DataFrame, col_names:list)->pd.DataFrame:
#     """
#     Interact with util.impute_mode to 
#     1> Replace the missing value with its mean
#     2> Create a new colume *_mean to indicate there was a missing value

#     Parameters:
#         col_name (list) and dataframe (pd.DataFrame): Specify the column names within dataframe for the function perform processing of mean.
        
#     Returns:
#         dataframe (pd.DataFrame): Return back the processed dataset    
#     """ 
#     for col_name in col_names:
#         dataframe = util.impute_mean(dataframe,col_name)
#     return dataframe

# def fe_impute_median(dataframe:pd.DataFrame, col_names:list)->pd.DataFrame:
#     """
#     Interact with util.impute_mode to 
#     1> Replace the missing value with its median
#     2> Create a new colume *_median to indicate there was a missing value

#     Parameters:
#         col_name (list) and dataframe (pd.DataFrame): Specify the column names within dataframe for the function perform processing of median.
        
#     Returns:
#         dataframe (pd.DataFrame): Return back the processed dataset    
#     """ 
#     for col_name in col_names:
#         dataframe = util.impute_mean(dataframe,col_name)
#     return dataframe

# def fe_impute_random(dataframe:pd.DataFrame, col_names:list)->pd.DataFrame:
#     """
#     Interact with util.impute_rand to 
#     1> Replace the missing value with its random
#     2> Create a new colume *_rand to indicate there was a missing value

#     Parameters:
#         col_name (list) and dataframe (pd.DataFrame): Specify the column names within dataframe for the function perform processing of random.
        
#     Returns:
#         dataframe (pd.DataFrame): Return back the processed dataset    
#     """ 
#     for col_name in col_names:
#         dataframe = util.impute_random(dataframe,col_name)
#     return dataframe

# def convert_columns_to_numeric(dataframe: pd.DataFrame)-> pd.DataFrame:
#     """
#     Perform feature_engineering on the following columns:
#         a> Convert columns with binary values to numeric 
#             i> Gender - Female to 0 and Male to 1 
#             ii> Cruise Name - Blastoise to 0 and Lapras to 1
#         b> Transform DOB to another new column Age
#         c> Perform Ordinal Encoding on 
#             i> Onboard Wifi Service
#             ii> Onboard Dining Service
#             iii> Onboard Entertainment"
#         d> Perform Label Encode on "Ticket Type"
#         e> Perform One hot Key Encode on "Source of Traffic"
#     Parameters:
#         dataframe (pd.DataFrame): The dataset in which the feature enginneering will be acting on

#     Returns:
#         dataframe (pd.DataFrame): Return back the processed dataset
#     """
#     dataframe = fe_convert_binary_columns(dataframe,"Gender",["Female","Male"])
#     dataframe = fe_convert_binary_columns(dataframe,"Cruise Name",["Blastoise","Lapras"])
#     dataframe = fe_age(dataframe)
#     dataframe = fe_ordinal_encode(dataframe,["Onboard Wifi Service","Onboard Dining Service","Onboard Entertainment"])
#     dataframe = util.remove_record_with_missing_data(dataframe,"Ticket Type")
#     dataframe = util.remove_duplicates_keeping_last(dataframe,"Ext_Intcode","index")
#     dataframe = fe_label_encode(dataframe, "Ticket Type")
#     dataframe = fe_one_hot_key(dataframe, ["Source of Traffic"])
#     return dataframe

# def impute_missing_data(dataframe:pd.DataFrame)->pd.DataFrame:
#     """
#     Perform impute on missing data

#     Parameters:
#         col_name (str) and dataframe (pd.DataFrame): Specify the column name within dataframe for the function perform processing.
        

#     Returns:
#         dataframe (pd.DataFrame): Return back the processed dataset
#     """ 
#     dataframe= fe_impute_mode(dataframe,["Onboard Wifi Service","Embarkation/Disembarkation time convenient", 
#                               "Ease of Online booking", "Gate location","Onboard Dining Service","Online Check-in","Cabin Comfort",
#                               "Onboard Entertainment","Cabin service","Baggage handling","Port Check-in Service","Onboard Service",
#                               "Cleanliness","Cruise Name"])
#     dataframe=fe_impute_mean(dataframe,["Distance in KM"])
#     dataframe=fe_impute_median(dataframe,["Age"])
#     dataframe=fe_impute_random(dataframe, ["Gender"])

#     return dataframe