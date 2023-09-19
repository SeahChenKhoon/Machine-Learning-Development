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

    def remove_invalid_data_in_datetime_col (self, dataframe:pd.DataFrame, column_name:str)->pd.DataFrame:
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

    def convert_datatype_object_to_numeric_col (self, dataframe: pd.DataFrame, feature: str)->pd.Series:
        """
        Summary:
            This function convert the specific variable column from Object to Numeric Data Type
        """
        dataframe[feature] = pd.to_numeric(dataframe[feature], errors='coerce')

    def convert_miles_to_km(self, dataframe:pd.DataFrame, col_name:str, new_fields: list)->pd.DataFrame:
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
        dataframe[new_fields] = dataframe[col_name].str.split(pat=' ', n=1, expand=True)
        self.convert_datatype_object_to_numeric_col(dataframe,new_fields[0])

        # 2> Convert the distance into KM.
        conversion_factors = {'Miles': 1.60934, 'KM': 1.0}
        dataframe[new_fields[0]] = abs(round(dataframe[new_fields[0]] * dataframe[new_fields[1]].map(conversion_factors),0))
        
        # 3> Drop "Cruise Distance" and "UOM" columns from dataset
        dataframe.drop(new_fields[1],axis=1,inplace=True)
        dataframe.drop(col_name,axis=1,inplace=True)    
        return dataframe
        
    def convert_miles_to_km_dummy(self, dataframe:pd.DataFrame, col_name:str)->None:
        """
            This function performs the following using the column_name and dataframe:
            1> Strip the distance away from the unit into 2 new columns "Distance" and "UOM". 
            2> Convert the distance into KM into "Distance in KM"
            3> Drop "Cruise Distance" and "UOM" columns from dataset
        Parameters:
            col_name (str) and dataframe (pd.DataFrame): 
            Specify the column name within dataframe for the function perform processing.
            
        Returns:
            dataframe (pd.DataFrame): 
        """
        # 1> Strip the distance away from the unit into 2 new columns "Distance" and "UOM"
        dataframe[["Distance", "UOM"]] = dataframe["Cruise Distance"].str.split(pat=' ', n=1, expand=True)
        # convert_text_to_numeric_col(dataframe,'Distance')
        # # 2> Convert the distance into KM.
        # conversion_factors = {'Miles': 1.60934, 'KM': 1.0}
        # dataframe['Distance in KM'] = abs(round(dataframe['Distance'] * dataframe['UOM'].map(conversion_factors),0))
        # dataframe.drop("Distance",axis=1,inplace=True)
        # dataframe.drop("UOM",axis=1,inplace=True)
        # dataframe.drop("Cruise Distance",axis=1,inplace=True)    
        return dataframe