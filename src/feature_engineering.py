import pandas as pd
import impute_missing_data
import util
import data_preprocessing
import convert_features_to_numeric
import modelling

class FeatureEngineer:
    def feature_grouping(self, merged_data:pd.DataFrame, column_list:list[str], column_grp:str):
        mean_val = merged_data[column_list].mean(axis=1).round()
        merged_data[column_grp] = mean_val
        util.drop_columns(merged_data, column_list)
