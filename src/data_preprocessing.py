import sqlite3
import datetime
import pandas as pd

class ReadData:
    def __init__(self, datapath):
        self._datapath = datapath

    def _read_data_from_db_file(self, db_file_path:str, table_name:str, index_col:str=None)->pd.DataFrame:
        """
        Read the .db file into a dataset with index column if provided.

        Parameters:
            db_file_path (str): The filename and path in which the process to read from
            table_name (str): The table name in which the function to read as
            index (str): Specify the index column of the dataset  
        Returns:
            dataframe (pd.DataFrame): Return back the processed dataset
        """
        try:
            connection = sqlite3.connect(db_file_path + table_name + ".db")
            df_cruise_pre = pd.read_sql_query("SELECT * FROM " + table_name, connection)
            if index_col != None:
                df_cruise_pre = df_cruise_pre.set_index(index_col)
            return df_cruise_pre
        except sqlite3.Error as e:
            print("SQLite error:", e)
            return None

    def read_data(self)->pd.DataFrame:
        """
        Perform the following:
        * Read pre-cruise data into df_cruise_pre dataset
        * Read post-cruise data into df_cruise_post dataset
        * Combine both pre-cruise and post-cruise dataset together 
        Parameters:
            None
        Returns:
            df_cruise (pd.DataFrame): Return back the combined datasets of pre-cruise and post-cruise
        """
        df_cruise_pre =  self._read_data_from_db_file(self._datapath, "cruise_pre","index")
        df_cruise_post =  self._read_data_from_db_file(self._datapath, "cruise_post","index")
        df_cruise = pd.merge(df_cruise_pre, df_cruise_post, on='Ext_Intcode', how='inner')
        return df_cruise


class DataPreprocessing:
    def process_data_preprocessing (self, dataframe:pd.DataFrame)->pd.DataFrame:
        """
        This function carries out preprocessing by 
        1> Date of Birth - Remove invalid_data=
        2> Cruise Distance - Convert Miles to KM 
        3> Ticket Type  - Remove rows with missing data

        Parameters:
            dataframe (pd.DataFrame): Specify the column name within dataframe for the function perform processing.
            
        Returns:
            dataframe (pd.DataFrame): Return back the processed dataset
        """
        dataframe = self._remove_invalid_data_in_datetime_col(dataframe,"Date of Birth")
        dataframe = self._convert_miles_to_km(dataframe)
        dataframe = self._remove_rows_with_missing_data(dataframe,"Ticket Type")
        dataframe = self._convert_DOB_to_age(dataframe,"Date of Birth")
        return dataframe
    
    def _convert_DOB_to_age(self, dataframe:pd.DataFrame, col_name:str)->pd.DataFrame:
        """
        Convert DOB to additional Age column

		Parameters:
			col_name (str) and dataframe (pd.DataFrame): Specify the column name within dataframe for the function perform processing.

		Returns:
			dataframe (pd.DataFrame): Return back the processed dataset
        """
        current_year = datetime.datetime.now().year
        dataframe["Age"] = current_year - dataframe[col_name].dt.year
        return dataframe


    def _remove_rows_with_missing_data (self, dataframe:pd.DataFrame, col_name:str)->pd.DataFrame:
        """
        This function will remove all rows with missing data on the specific col_name within the dataframe

        Parameters:
            col_name (str) and dataframe (pd.DataFrame): Specify the column name within dataframe for the function perform processing.
            
        Returns:
            dataframe (pd.DataFrame): Return back the processed dataset
        """
        dataframe.dropna(subset=[col_name], inplace=True)
        return dataframe

    def _convert_miles_to_km(self, dataframe:pd.DataFrame)->pd.DataFrame:
        """
        This function performs the standardisation of UOM in Cruise Distance:
            1> Strip the distance away from the unit into 2 new columns "Distance" and "UOM". 
            2> Convert the distance into KM into "Distance in KM"
            3> Drop "Cruise Distance" and "UOM" columns from dataset
        Parameters:
            col_name (str) and dataframe (pd.DataFrame): Specify the column name within dataframe for the function perform processing.
            
        Returns:
            dataframe (pd.DataFrame):         
        """
        dataframe[["Distance in KM", "UOM"]] = dataframe["Cruise Distance"].str.split(pat=' ', n=1, expand=True)
        self._convert_datatype_object_to_numeric_col(dataframe,'Distance in KM')

        # 2> Convert the distance into KM.
        dataframe.loc[dataframe["UOM"]== "Miles","Distance in KM"] = dataframe["Distance in KM"] * 1.60934
        
        # 3> Drop "Cruise Distance" and "UOM" columns from dataset
        dataframe = self._drop_column(dataframe, ["UOM","Cruise Distance"])

        return dataframe
    
    def _remove_invalid_data_in_datetime_col (self, dataframe:pd.DataFrame, column_name:str)->pd.DataFrame:
        """
        This function removes all rows that have invalid datetime give the dataframe and column name

        Parameters:
            col_name (str) and dataframe (pd.DataFrame): Specify the column name within dataframe for the function perform processing.
            

        Returns:
            dataframe (pd.DataFrame): Return back the processed dataset
            
        """
        try:
            dataframe[column_name] = pd.to_datetime(dataframe[column_name], format='%d/%m/%Y', errors='coerce')
            dataframe = dataframe.dropna(subset=[column_name])
        except ValueError:
            print(f"Column '{column_name}' does not contain valid 'DD/MM/YYYY' dates.")
        return dataframe

    def _convert_datatype_object_to_numeric_col (self,dataframe: pd.DataFrame, feature: str)->pd.Series:
        """
        Convert the specific variable column from Object to Numeric Data Type

        Parameters:
            col_name (str) and dataframe (pd.DataFrame): Specify the column name within dataframe for the function perform processing.
            
        Returns:
            dataframe (pd.DataFrame): Return back the processed dataset
        """
        dataframe[feature] = pd.to_numeric(dataframe[feature], errors='coerce')

    def _drop_column(self, dataframe: pd.DataFrame, col_names: list)->pd.DataFrame:
        """
        Perform Removal duplicate records while retaining the last occurrence..

        Parameters:
            col_names (list) and dataframe (pd.DataFrame): Specify the column names within dataframe for the function perform processing.
            
        Returns:
            dataframe (pd.DataFrame): Return back the processed dataset
        """
        for col_name in col_names:
            dataframe.drop(col_name,axis=1,inplace=True)
        return dataframe

