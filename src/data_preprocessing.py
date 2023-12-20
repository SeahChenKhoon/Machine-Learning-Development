import pandas as pd
import util
import numpy as np
import ast
from sklearn.preprocessing import LabelEncoder

class DataProcessing:
    # def __init__(self, dataframe:pd.DataFrame, display_stub) -> None:
    #     self.dataframe = dataframe
    #     self.__display_stub = display_stub

    def get_dataframe(self) -> pd.DataFrame:
        return self.dataframe

    def impute_missing_value_info (self, impute_missing_val_info)->None:
        for impute_missing_val in impute_missing_val_info:
            self.impute_missing_value(ast.literal_eval(impute_missing_val['col_list']), impute_missing_val['impute_type'])

    def impute_missing_value(self,col_list, impute_type):
        if impute_type == "random":
            # Set a seed for reproducibility
            np.random.seed(42)
            for col_name in col_list:
                unique_values = self.dataframe[col_name].dropna().unique()
                self.dataframe[col_name] = self.dataframe[col_name].apply(lambda x: np.random.choice(unique_values) if pd.isnull(x) else x)
        elif impute_type=="mode":
            for col_name in col_list:
                mode_gender = self.dataframe[col_name].mode().iloc[0]
                self.dataframe[col_name].fillna(mode_gender, inplace=True)
        elif impute_type=="mean":
            for col_name in col_list:
                self.dataframe[col_name].fillna(self.dataframe[col_name].mean(), inplace=True)
        return None

    def label_encoder(self, list_cols: list) -> None:
        label_encoder = LabelEncoder()
        
        for col in ast.literal_eval(list_cols):
            self.dataframe[col] = label_encoder.fit_transform(self.dataframe[col].astype(str))
        if self.__display_stub == True:
            print(self.dataframe.shape)
        return None
    
    def numeric_conversion(self, numeric_field_info:list)->None:
        for numeric_field_info in numeric_field_info:
            self.convert_number(ast.literal_eval(numeric_field_info['col_list']), 
                                 numeric_field_info['dtype'])
        if self.__display_stub == True:
            print(self.dataframe.shape)
        return None

    def convert_number(self, col_list_info: list, dtype:str):
        for col_name in col_list_info:
            if dtype == 'int32':
                self.dataframe[col_name] = self.dataframe[col_name].astype('Int32')
            elif dtype == 'float64':
                self.dataframe[col_name] = self.dataframe[col_name].astype('Float64')
        return None


    def valid_data_processing(self, valid_data_info:list)->None:
        for valid_data_col in valid_data_info:
            self.restrict_val(ast.literal_eval(valid_data_col['col_list']), 
                                 ast.literal_eval(valid_data_col['valid_data_list']))
        if self.__display_stub == True:
            print(self.dataframe.shape)
        return None

    def restrict_val (self,  col_list:list[str], valid_val_list:list):
        for col_name in col_list:
            col_dtype = self.dataframe[col_name].dtype
            if self.__display_stub == True:
                print(col_name)
                print(col_dtype)
                print(self.dataframe[col_name].unique())
            self.dataframe = self.dataframe[self.dataframe[col_name].isin(valid_val_list)]
            if self.__display_stub == True:
                print(self.dataframe.shape)
        return None

    def split_composite_field(self,  dataframe, composite_fields:list)->None:
        if composite_fields:
            for composite_field in composite_fields:
                self.split_col(dataframe,composite_field['composite_field'], ast.literal_eval(composite_field['new_column_list']), 
                            composite_field['delimiter'])
        return dataframe

    def split_col(self, dataframe, composite_col: str, list_cols: list, delimiter: str):
        # Split the composite column into a list of values
        split_values = dataframe[composite_col].str.split(delimiter)

        # Create new columns from the list of values
        for i, new_col in enumerate(list_cols):
            dataframe[new_col] = split_values.str[i]

        # Drop the original composite column
        dataframe.drop(columns=[composite_col], inplace=True)
        return dataframe
    
    def dirty_data_processing(self, dirty_data_info:list)->None:
        for dirty_data in dirty_data_info:
            self.replace_value(ast.literal_eval(dirty_data['field_list']), dirty_data['replace_val'],  
                                   dirty_data['replace_with'], dirty_data['like_ind'])
        if self.__display_stub == True:
            print(self.dataframe.shape)
        return None
    
    def replace_value (self, col_list:list[str], replace_val:any, replace_with:any,
                        like_ind:bool=False):
        for col_name in col_list:
            if like_ind == False:
                if replace_with == "None":
                    replace_with = None
                self.dataframe.loc[self.dataframe[col_name]==replace_val,col_name] = replace_with
            else:
                str_len = len(replace_val)
                self.dataframe['substring'] = self.dataframe[col_name].str[:str_len]
                self.dataframe.loc[self.dataframe['substring'].str.upper()== replace_val.upper(), col_name] = replace_with
                util.util_rm_col(self.dataframe,'substring')
        if self.__display_stub == True:
            print(self.dataframe.shape)
        return None 

    def replace_nan_none(self, df)->None:
        df.replace({np.nan: None},inplace=True)
        return df

    def rm_cols_high_missing(self, threshold)->None:
        # Calculate the percentage of missing values for each column
        missing_percentages = self.dataframe.isnull().mean()
        # Identify columns exceeding the threshold
        columns_to_remove = missing_percentages[missing_percentages > threshold].index
        self.dataframe.drop(columns=columns_to_remove, inplace=True)
        if self.__display_stub == True:
            print(self.dataframe.shape)
        return None

    def rm_rows_target_var(self, target_col: str) -> None:
        # Remove rows with missing values in target columns
        self.dataframe.dropna(subset=target_col, inplace=True)
        if self.__display_stub == True:
            print(self.dataframe.shape)
        return None

    def remove_missing(self, list_cols: list) -> None:
        # Remove rows with missing values in specified columns
        self.dataframe.dropna(subset=list_cols, inplace=True)
        if self.__display_stub == True:
            print(self.dataframe.shape)
        return None

    def obj_to_datetime(self, datetime_fields_info:list)->None:
        if datetime_fields_info:
            for datetime_field_info in datetime_fields_info:
                col_names = ast.literal_eval(datetime_field_info['column_list'])
                for col_name in col_names:
                    self.dataframe[col_name] = pd.to_datetime(self.dataframe[col_name], format=datetime_field_info['format'], errors='coerce')
        if self.__display_stub == True:
            print(self.dataframe.shape)
        return None

    def rm_id_cols(self, dataframe, list_cols:list[str]):
        util.util_rm_col(dataframe, list_cols)
        return dataframe

    def yyyy_from_date(self, date_yyyy_info:list)->None:
        self.convert_datetime_to_year(ast.literal_eval(date_yyyy_info['col_list']), ast.literal_eval(date_yyyy_info['yyyy_col_list']))
        if self.__display_stub == True:
            print(self.dataframe.shape)
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
