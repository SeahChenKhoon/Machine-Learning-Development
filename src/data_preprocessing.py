import pandas as pd
import util
import numpy as np
import ast
from sklearn.preprocessing import LabelEncoder

class DataProcessing:
    def get_dataframe(self) -> pd.DataFrame:
        return self.dataframe

    def impute_missing_value_info (self, df, impute_missing_val_info)->None:
        for impute_missing_val in impute_missing_val_info:
            df = self.impute_missing_value(df, ast.literal_eval(impute_missing_val['col_list']), impute_missing_val['impute_type'])
        return df

    def impute_missing_value(self,df, col_list, impute_type):
        if impute_type == "random":
            # Set a seed for reproducibility
            np.random.seed(42)
            for col_name in col_list:
                unique_values = df[col_name].dropna().unique()
                df[col_name] = df[col_name].apply(lambda x: np.random.choice(unique_values) if pd.isnull(x) else x)
        elif impute_type=="mode":
            for col_name in col_list:
                mode_gender = df[col_name].mode().iloc[0]
                df[col_name].fillna(mode_gender, inplace=True)
        elif impute_type=="mean":
            for col_name in col_list:
                df[col_name].fillna(df[col_name].mean(), inplace=True)
        return df

    def label_encoder(self, df, list_cols: list) -> None:
        label_encoder = LabelEncoder()
        
        for col in ast.literal_eval(list_cols):
            df[col] = label_encoder.fit_transform(df[col].astype(str))
        return df
    
    def numeric_conversion(self, df, numeric_field_info:list)->None:
        for numeric_field_info in numeric_field_info:
            self.convert_number(df, ast.literal_eval(numeric_field_info['col_list']), 
                                 numeric_field_info['dtype'])
        return df

    def convert_number(self, df, col_list_info: list, dtype:str):
        for col_name in col_list_info:
            if dtype == 'int32':
                df[col_name] = df[col_name].astype('Int32')
            elif dtype == 'float64':
                df[col_name] = df[col_name].astype('Float64')
        return df


    def valid_data_processing(self, df, valid_data_info:list)->pd.DataFrame:
        for valid_data_col in valid_data_info:
            df = self.restrict_val(df, ast.literal_eval(valid_data_col['col_list']), 
                                 ast.literal_eval(valid_data_col['valid_data_list']))
        return df

    def restrict_val (self, df, col_list:list[str], valid_val_list:list):
        for col_name in col_list:
            col_dtype = df[col_name].dtype
            df = df[df[col_name].isin(valid_val_list)]
        return df

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
    
    def dirty_data_processing(self, df, dirty_data_info:list)->None:
        for dirty_data in dirty_data_info:
            self.replace_value(df, ast.literal_eval(dirty_data['field_list']), dirty_data['replace_val'],  
                                   dirty_data['replace_with'], dirty_data['like_ind'])
        return df
    
    def replace_value (self, df, col_list:list[str], replace_val:any, replace_with:any,
                        like_ind:bool=False):
        for col_name in col_list:
            if like_ind == False:
                if replace_with == "None":
                    replace_with = None
                df.loc[df[col_name]==replace_val,col_name] = replace_with
            else:
                str_len = len(replace_val)
                df['substring'] = df[col_name].str[:str_len]
                df.loc[df['substring'].str.upper()== replace_val.upper(), col_name] = replace_with
                util.util_rm_col(df,'substring')
        return df 

    def replace_nan_none(self, df)->None:
        df.replace({np.nan: None},inplace=True)
        return df

    def rm_cols_high_missing(self, df, threshold)->None:
        # Calculate the percentage of missing values for each column
        missing_percentages = df.isnull().mean()
        # Identify columns exceeding the threshold
        columns_to_remove = missing_percentages[missing_percentages > threshold].index
        df.drop(columns=columns_to_remove, inplace=True)
        return df

    def rm_rows_target_var(self, df, target_col: str) -> None:
        # Remove rows with missing values in target columns
        df.dropna(subset=target_col, inplace=True)
        return df

    def remove_missing(self, df, list_cols: list) -> None:
        # Remove rows with missing values in specified columns
        df.dropna(subset=list_cols, inplace=True)
        return df

    def obj_to_datetime(self, df, datetime_fields_info:list)->None:
        if datetime_fields_info:
            for datetime_field_info in datetime_fields_info:
                col_names = ast.literal_eval(datetime_field_info['column_list'])
                for col_name in col_names:
                    df[col_name] = pd.to_datetime(df[col_name], format=datetime_field_info['format'], errors='coerce')
        return df

    def rm_id_cols(self, df, list_cols:list[str]):
        util.util_rm_col(df, list_cols)
        return df


