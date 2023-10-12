import pandas as pd
import util

class FeatureEngineer:
    def feature_grouping(self, merged_data:pd.DataFrame, column_list:list[str], column_grp:str)->None:
        """
        Perform feature grouping on the input DataFrame.

        Parameters:
            merged_data (pd.DataFrame): The input DataFrame.
            column_list (list[str]): List of column names to calculate the mean from.
            column_grp (str): Name of the new column to be added.

        Returns:
            None
        """
        mean_val = merged_data[column_list].mean(axis=1).round()
        merged_data[column_grp] = mean_val
        util.drop_columns(merged_data, column_list)
        return
