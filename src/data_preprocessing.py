import sqlite3
import pandas as pd

class DataPreprocessing:
    def __init__(self, datapath):
        self.datapath = datapath

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
        df_cruise_pre =  self._read_data_from_db_file(self.datapath, "cruise_pre","index")
        df_cruise_post =  self._read_data_from_db_file(self.datapath, "cruise_post","index")
        df_cruise = pd.merge(df_cruise_pre, df_cruise_post, on='Ext_Intcode', how='inner')
        return df_cruise

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

    def fix_typo_error(self, dataframe:pd.DataFrame, col_name:str, 
                    replace_list:list, replace_with:str) -> pd.DataFrame:
        """
        Perform fixing of typo error list of data (replace_list) to replace_with 

        Parameters:
            col_name (str) and dataframe (pd.DataFrame): Specify the column name within dataframe for the function perform processing.
            replace_list (list): List of element to be replace
            replace_with (str): String to replace the element in replace_list

        Returns:
            dataframe (pd.DataFrame): Return back the processed dataset 
        """
        for word in replace_list:
            dataframe.loc[dataframe[col_name]==word,col_name] = replace_with
        return dataframe

