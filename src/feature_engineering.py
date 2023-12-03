import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
import util



def fe_SMOTE(X, y, random_state):
    os = SMOTE(random_state=random_state)
    columns = X.columns
    return os.fit_resample(X, y)

def convert_miles_to_KM (df_dataframe: pd.DataFrame, col_name:str)->None:
    df_dataframe[col_name] *= 1.609344
    return None

def calc_year_diff (df_dataframe: pd.DataFrame, minuend_col:str, subtrahend_col:str, new_col:str)->None:
    df_dataframe[new_col] = df_dataframe[minuend_col] - df_dataframe[subtrahend_col]
    util.remove_col(df_dataframe, [minuend_col, subtrahend_col])
    return None

def denote_missing_col(df_dataframe):
    missing_col_list = []
    for col_name in df_dataframe.columns:
        if df_dataframe[col_name].isnull().sum() > 0:
            # df_dataframe[col_name + "_missing"] = df_dataframe[col_name].fillna(11)
            df_dataframe[col_name + "_missing"] = df_dataframe[col_name].isnull().astype(int)
            missing_col_list.append(col_name + "_missing")
    print(missing_col_list)
    # Create a new column "tot_missing_col" that represents the total number of missing values
    df_dataframe['tot_missing_col'] = df_dataframe[missing_col_list].sum(axis=1)
    print(df_dataframe['tot_missing_col'].head())
    util.remove_col(df_dataframe, missing_col_list)
    return None
        

def impute_missing_value(df_dataframe: pd.DataFrame, impute_type, col_list:list[str]=None,none_val=None)->None:
    if impute_type =="mean":
        # Iterate over columns
        for col_name in df_dataframe.columns[df_dataframe.isna().any()].tolist():
            # Calculate the mean for the column
            mean_value = round(df_dataframe[col_name].mean(),0)
            
            # Impute missing values in the column with the mean
            df_dataframe[col_name].fillna(mean_value, inplace=True)
    elif impute_type == "random":
        # Set a seed for reproducibility
        np.random.seed(42)
        for col_name in col_list:
            if none_val != None:
                df_dataframe[col_name] = df_dataframe[col_name].replace(2, None)
                # Create a mask of missing values in the column
                missing_mask = df_dataframe[col_name].isnull()

                # Generate random values to fill missing values
                random_values = np.random.choice([0, 1], size=np.sum(missing_mask))

                # Impute the missing values with random values
                df_dataframe.loc[missing_mask, col_name] = random_values
    return None

# class FeatureEngineer:
#     def feature_grouping(self, merged_data:pd.DataFrame, column_list:list[str], column_grp:str)->None:
#         """
#         Perform feature grouping on the input DataFrame.

#         Parameters:
#             merged_data (pd.DataFrame): The input DataFrame.
#             column_list (list[str]): List of column names to calculate the mean from.
#             column_grp (str): Name of the new column to be added.

#         Returns:
#             None
#         """
#         mean_val = merged_data[column_list].mean(axis=1).round()
#         merged_data[column_grp] = mean_val
#         util.drop_columns(merged_data, column_list)
#         return
