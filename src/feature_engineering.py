import pandas as pd
import numpy as np

class ImputeMissingValue:
    def impute_missing_data(self,dataframe:pd.DataFrame):
        dataframe= self._fe_impute_mode(dataframe,["Onboard Wifi Service","Embarkation/Disembarkation time convenient", 
                                "Ease of Online booking", "Gate location","Onboard Dining Service","Online Check-in","Cabin Comfort",
                                "Onboard Entertainment","Cabin service","Baggage handling","Port Check-in Service","Onboard Service",
                                "Cleanliness","Cruise Name"])
        dataframe=self._fe_impute_median(dataframe,["Age"])
        dataframe=self._fe_impute_mean(dataframe,["Distance in KM"])
        dataframe=self._fe_impute_random(dataframe, ["Gender"])

    def _fe_impute_random(self,dataframe:pd.DataFrame, col_names:list)->pd.DataFrame:
        """
        Interact with util.impute_rand to 
        1> Replace the missing value with its random
        2> Create a new colume *_rand to indicate there was a missing value

        Parameters:
            col_name (list) and dataframe (pd.DataFrame): Specify the column names within dataframe for the function perform processing of random.
            
        Returns:
            dataframe (pd.DataFrame): Return back the processed dataset    
        """ 
        for col_name in col_names:
            dataframe = self._impute_random(dataframe,col_name)
        return dataframe

    def _impute_random(self, dataframe, col_name, add_mark_col=False)->pd.DataFrame:
        """
        Perform the following based on the specified column name and dataset 
        1> the creation of new column col_name + "_random" to track missing value indication
        2> Impute the missing value with its random value within the existing value range

        Parameters:
            col_name (str) and dataframe (pd.DataFrame): Specify the column name within dataframe for the function perform processing.
            

        Returns:
            dataframe (pd.DataFrame): Return back the processed dataset
        """
        dataframe[col_name + "_rand"] = np.where(dataframe[col_name].isnull(),1,0)
        random_values = dataframe[col_name].dropna().sample(dataframe[col_name].isna().sum(), replace=True).values
        dataframe.loc[dataframe[col_name].isna(), col_name] = random_values
        return dataframe

    def _fe_impute_mean(self, dataframe:pd.DataFrame, col_names:list)->pd.DataFrame:
        """
        Interact with util.impute_mode to 
        1> Replace the missing value with its mean
        2> Create a new colume *_mean to indicate there was a missing value

        Parameters:
            col_name (list) and dataframe (pd.DataFrame): Specify the column names within dataframe for the function perform processing of mean.
            
        Returns:
            dataframe (pd.DataFrame): Return back the processed dataset    
        """ 
        for col_name in col_names:
            dataframe = self._impute_mean(dataframe,col_name)
        return dataframe

    def _impute_mean(self,dataframe:pd.DataFrame, col_name, add_mark_col=False)->pd.DataFrame:
        """
        Perform the following based on the specified column name and dataset 
        1> the creation of new column col_name + "_mean" to track missing value indication
        2> Impute the missing value with its mean value 

        Parameters:
            col_name (str) and dataframe (pd.DataFrame): Specify the column name within dataframe for the function perform processing.
            

        Returns:
            dataframe (pd.DataFrame): Return back the processed dataset
        """
        if add_mark_col:
            dataframe[col_name + "_mean"] = np.where(dataframe[col_name].isnull(),1,0)
        dataframe[col_name] = dataframe[col_name].fillna(int(dataframe[col_name].mean()))
        return dataframe

    def _fe_impute_median(self,dataframe:pd.DataFrame, col_names:list)->pd.DataFrame:
        """
        Interact with util.impute_mode to 
        1> Replace the missing value with its median
        2> Create a new colume *_median to indicate there was a missing value

        Parameters:
            col_name (list) and dataframe (pd.DataFrame): Specify the column names within dataframe for the function perform processing of median.
            
        Returns:
            dataframe (pd.DataFrame): Return back the processed dataset    
        """ 
        for col_name in col_names:
            dataframe = self.impute_mean(dataframe,col_name)
        return dataframe

    def _fe_impute_median(self,dataframe:pd.DataFrame, col_name, add_mark_col=False)->pd.DataFrame:
        """
        Perform Imputation of specific column in dataset to median

		Parameters:
			col_name (str) and dataframe (pd.DataFrame): Specify the column name within dataframe for the function perform processing.
			

		Returns:
			dataframe (pd.DataFrame): Return back the processed dataset
        """
        if add_mark_col:
            dataframe[col_name + "_median"] = np.where(dataframe[col_name].isnull(),1,0)
        dataframe[col_name] = dataframe[col_name].fillna(int(dataframe[col_name].median()))
        return dataframe


    def _fe_impute_mode(self,dataframe:pd.DataFrame, col_names:list)->pd.DataFrame:
        """
        Pass each of the specific column in the list for imputation of mode
        
        Parameters:
            col_name (str) and dataframe (pd.DataFrame): Specify the column name within dataframe for the function perform processing.
            
        Returns:
            dataframe (pd.DataFrame): Return back the processed dataset
        """
        for col_name in col_names:
            dataframe = self._impute_mode(dataframe,col_name)
        return dataframe
    
    def _impute_mode(self, dataframe:pd.DataFrame, col_name:str, add_mark_col=False)->pd.DataFrame:
        """
        Perform the following based on the specified column name and dataset 
        1> the creation of new column col_name + "_mode" to track missing value indication
        2> Impute the missing value with its mode

        Parameters:
            col_name (str) and dataframe (pd.DataFrame): Specify the column name within dataframe for the function perform processing.
            
        Returns:
            dataframe (pd.DataFrame): Return back the processed dataset
        """
        if add_mark_col:
            dataframe[col_name + "_mode"] = np.where(dataframe[col_name].isnull(),1,0)
        dataframe[col_name] = dataframe[col_name].fillna(dataframe[col_name].mode()[0])
        return dataframe
    
