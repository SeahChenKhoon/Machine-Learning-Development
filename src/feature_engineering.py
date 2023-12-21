import pandas as pd
import numpy as np
import seaborn as sns
import ast
import matplotlib.pyplot as plt
import util

class feature_engineering:
    def __init__(self, dataframe:pd.DataFrame) -> None:
        self.dataframe = dataframe
        return None

    def yyyy_from_date(self, date_yyyy_info:list)->None:
        self.convert_datetime_to_year(ast.literal_eval(date_yyyy_info['col_list']), ast.literal_eval(date_yyyy_info['yyyy_col_list']))
        return None

    def convert_datetime_to_year(self, list_cols:list[str], list_new_cols:list)->None:
        count =0
        for col_name in list_cols:
            new_col = list_new_cols[count]
            self.dataframe[new_col] = self.dataframe[col_name].dt.year.astype(np.int32)
            count += 1
        util.util_rm_col(self.dataframe, list_cols)
        if self.__display_stub == True:
            print(self.dataframe.shape)
        return None
    
    def get_dataframe(self) -> pd.DataFrame:
        return self.dataframe

    def one_hot_key_encode(self, col_names:list[str]):
        self.dataframe = pd.get_dummies(self.dataframe, columns=col_names, drop_first=True)
        return None

    def convert_miles_to_KM (self,col_name:str)->None:
        self.dataframe[col_name] = abs(self.dataframe[col_name] * 1.609344)
        return None

    def calc_year_diff (self, minuend_col:str, subtrahend_col:str, new_col:str)->None:
        self.dataframe[new_col] = self.dataframe[minuend_col] - self.dataframe[subtrahend_col]
        util.util_rm_col(self.dataframe, [minuend_col, subtrahend_col])
        return None

    def denote_missing_col(self):
        missing_col_list = []
        for col_name in self.dataframe.columns:
            if self.dataframe[col_name].isnull().sum() > 0:
                self.dataframe[col_name + "_missing"] = self.dataframe[col_name].isnull().astype(int)
                missing_col_list.append(col_name + "_missing")
        # Create a new column "tot_missing_col" that represents the total number of missing values
        self.dataframe['tot_missing_col'] = self.dataframe[missing_col_list].sum(axis=1)
        util.util_rm_col(self.dataframe, missing_col_list)
        return None
            

    def impute_missing_value(self, impute_type, col_list:list[str]=None,none_val=None)->None:
        if impute_type =="mean":
            # Iterate over columns
            for col_name in self.dataframe.columns[self.dataframe.isna().any()].tolist():
                # Calculate the mean for the column
                mean_value = round(self.dataframe[col_name].mean(),0)
                
                # Impute missing values in the column with the mean
                self.dataframe[col_name].fillna(mean_value, inplace=True)
        elif impute_type == "random":
            # Set a seed for reproducibility
            np.random.seed(42)
            for col_name in col_list:
                if none_val != None:
                    self.dataframe[col_name] = self.dataframe[col_name].replace(2, None)
                    # Create a mask of missing values in the column
                    missing_mask = self.dataframe[col_name].isnull()

                    # Generate random values to fill missing values
                    random_values = np.random.choice([0, 1], size=np.sum(missing_mask))

                    # Impute the missing values with random values
                    self.dataframe.loc[missing_mask, col_name] = random_values
        return None
