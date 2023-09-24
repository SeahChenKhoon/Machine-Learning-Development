import pandas as pd
import numpy as np

class ImputeMissingData:
    def __init__(self, dataframe:pd.DataFrame, feature_list:list)->None:
        self._dataframe = dataframe
        self._feature_list = feature_list
        self._add_mark_col = False

class ImputeMean(ImputeMissingData):
    def process_impute(self):
        """
        Process imputation of mean on missing data

        Parameters:
            None
        """
        for feature in self._feature_list:
            self._impute_mean(feature)
        return self._dataframe
    
    def _impute_mean(self,col_name:str)->pd.DataFrame:
        """
        Perform Imputation of specific column in dataset to mean

		Parameters:
			col_name (str): Specify the column name within dataframe for the function perform processing.
			
		Returns:
			dataframe (pd.DataFrame): Return back the processed dataset
        """
        if self._add_mark_col:
            self._dataframe[col_name + "_mean"] = np.where(self._dataframe[col_name].isnull(),1,0)
        self._dataframe[col_name] = self._dataframe[col_name].fillna(int(self._dataframe[col_name].mean()))
        return self._dataframe        

class ImputeMedian(ImputeMissingData):
    def process_impute(self):
        """
        Process imputation of media on missing data

        Parameters:
            None
        """
        for feature in self._feature_list:
            self._impute_median(feature)
        return self._dataframe
    
    def _impute_median(self,col_name:str)->pd.DataFrame:
        """
        Perform Imputation of specific column in dataset to median

		Parameters:
			col_name (str): Specify the column name within dataframe for the function perform processing.
			
		Returns:
			dataframe (pd.DataFrame): Return back the processed dataset
        """
        if self._add_mark_col:
            self._dataframe[col_name + "_median"] = np.where(self._dataframe[col_name].isnull(),1,0)
        self._dataframe[col_name] = self._dataframe[col_name].fillna(int(self._dataframe[col_name].median()))
        return self._dataframe        
