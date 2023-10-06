import sqlite3
import datetime
import util
import pandas as pd

# class ReadData:
#     def __init__(self, datapath):
#         self._datapath = datapath

#     def _read_data_from_db_file(self, db_file_path:str, table_name:str, index_col:str=None)->pd.DataFrame:
#         """
#         Read the .db file into a dataset with index column if provided.

#         Parameters:
#             db_file_path (str): The filename and path in which the process to read from
#             table_name (str): The table name in which the function to read as
#             index (str): Specify the index column of the dataset  
#         Returns:
#             dataframe (pd.DataFrame): Return back the processed dataset
#         """
#         try:
#             connection = sqlite3.connect(db_file_path + table_name + ".db")
#             df_cruise_pre = pd.read_sql_query("SELECT * FROM " + table_name, connection)
#             if index_col != None:
#                 df_cruise_pre = df_cruise_pre.set_index(index_col)
#             return df_cruise_pre
#         except sqlite3.Error as e:
#             print("SQLite error:", e)
#             return None

#     def read_data(self)->pd.DataFrame:
#         """
#         Perform the following:
#         * Read pre-cruise data into df_cruise_pre dataset
#         * Read post-cruise data into df_cruise_post dataset
#         * Combine both pre-cruise and post-cruise dataset together 
#         Parameters:
#             None
#         Returns:
#             df_cruise (pd.DataFrame): Return back the combined datasets of pre-cruise and post-cruise
#         """
#         df_cruise_pre =  self._read_data_from_db_file(self._datapath, "cruise_pre","index")
#         df_cruise_post =  self._read_data_from_db_file(self._datapath, "cruise_post","index")
#         df_cruise = pd.merge(df_cruise_pre, df_cruise_post, on='Ext_Intcode', how='inner')
#         return df_cruise


class DataPreprocessing:
    def __init__(self, data_path:str, tablename_1:str, tablename_2:str, index_col:str)->None:
        """
        Initialize a DataPreprocessing object.

        This constructor sets up the DataPreprocessing object with the necessary
        information to load and process data from two data files.

        Parameters:
            self (DataPreprocessing): : The instance of the class.
            data_path (str): The base path where the data files are located.
            tablename_1 (str): The name of the first table name to be processed.
            tablename_2 (str): The name of the second table name to be processed.
            index_col (str): The common column that joins the 2 data files

        Returns:
            None
        """
        self.data_path:str =  data_path
        self.tablename_1:str = tablename_1
        self.tablename_2:str = tablename_2
        self.index_col:str = index_col
        self.data1:pd.DataFrame = None
        self.data2:pd.DataFrame = None
        self.merged_data:pd.DataFrame = None

    def load_data(self)->None:
        """
        Load data from two SQLite database tables into Pandas DataFrames.

        This method calls the `read_db` method to read data from two specified
        SQLite database tables into Pandas DataFrames, and assigns them to
        instance variables `self.data1` and `self.data2`.

        Parameters:
            self (DataPreprocessing): The instance of the class.

        Returns:
            None
        """
        self.data1:pd.DataFrame = self.read_db(self.data_path, self.tablename_1,self.index_col)
        self.data2:pd.DataFrame = self.read_db(self.data_path, self.tablename_2,self.index_col)

    def read_db(self, db_data_path:str, table_name:str, index_col:str=None)->pd.DataFrame:
        """
        Read data from a SQLite database table into a Pandas DataFrame.

        This method connects to a SQLite database located at `db_data_path` and reads data
        from the specified `table_name` into a Pandas DataFrame. Optionally, you can specify
        an `index_col` to set as the DataFrame's index.

        Parameters:
            self (DataPreprocessing): The instance of the class.
            db_data_path (str): The path to the SQLite database file.
            table_name (str): The name of the table to read data from.
            index_col (str, optional): The column to set as the DataFrame's index (default: None).

        Returns:
            pd.DataFrame: A Pandas DataFrame containing the data from the specified table.
                        Returns None if an error occurs during database access.

        Raises:
            sqlite3.Error: If there's an error with the SQLite database connection or query.

        """
        try:
            connection = sqlite3.connect(db_data_path + table_name + ".db")
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

    def merge_data(self):
        """
        Merge two Pandas DataFrames based on a common index column.

        This method performs an inner join operation on the two Pandas DataFrames
        stored in `self.data1` and `self.data2`, using a common index column named
        'common_index_column'. The result is stored in the `self.merged_data`
        attribute.

        If either `self.data1` or `self.data2` is `None`, or if the common index
        column 'common_index_column' is not present in both DataFrames, no merging
        will occur, and a message is printed indicating that data should be loaded
        first using the `load_data` method.

        Parameters:
            self (DataPreprocessing): The instance of the class.

        Returns:
            None
        """
        if self.data1 is not None and self.data2 is not None:
            self.merged_data = pd.merge(self.data1, self.data2, on=self.index_col, how='inner')
        else:
            print("Data not loaded. Call load_data() first.")
        return

    def clean_datetime_col (self, column_name:str)->None:
        """
        Remove rows with invalid date values in a specified datetime column.

        This method takes a Pandas DataFrame and a column name containing date values
        in the format '%d/%m/%Y'. It converts the values in the specified column to
        datetime objects and drops rows with invalid or missing date values.

        Parameters:
            self (DataPreprocessing): The instance of the class.
            column_name (str): The name of the column with date values to be processed.

        Returns:
            None

        Notes:
            - The 'coerce' option in pd.to_datetime converts invalid dates to NaT (Not a Timestamp).
            - Rows with NaT values are dropped from the DataFrame. 
        """
        self.merged_data[column_name] = pd.to_datetime(self.merged_data[column_name], format='%d/%m/%Y', errors='coerce')
        self.merged_data = self.merged_data.dropna(subset=[column_name])
        return

    def mileage_conversion(self)->None:
        """
        Convert distance values from miles to kilometers in the merged DataFrame.

        This method extracts distance values from the 'Cruise Distance' column, splits them
        into distance and unit of measurement (UOM), converts distance values from miles
        to kilometers where necessary, and updates the 'Distance in KM' column in the
        merged DataFrame with the converted values. The 'Cruise Distance' and 'UOM' columns
        are dropped from the dataset.

        Parameters:
            self (DataPreprocessing): The instance of the class.

        Returns:
            None
        """
        # Split "Cruise Distance" into "Distance", "UOM" storing distance and and KM/Miles respectively
        self.merged_data[["Distance", "UOM"]] = self.merged_data["Cruise Distance"].str.split(pat=' ', n=1, expand=True)

        # Store numeric dataframe["Distance"] into a temporarily dataframe temp_dist["Distance in KM"] and then
        # convert to KM
        temp_dist = self.merged_data[(~self.merged_data["Distance"].isna())][["Distance","UOM"]]
        temp_dist["Distance in KM"] = pd.to_numeric(temp_dist['Distance'], errors='coerce')
        temp_dist.loc[temp_dist["UOM"]== "Miles","Distance in KM"] = temp_dist["Distance in KM"] * 1.60934
        temp_dist["Distance in KM"] = temp_dist["Distance in KM"].abs().round(0)
        temp_dist['Distance in KM'] = temp_dist['Distance in KM'].astype(int)

        self.merged_data.loc[~self.merged_data["Distance"].isnull(),"Distance in KM"] =  temp_dist["Distance in KM"]
        
        # 3> Drop "Cruise Distance" and "UOM" columns from dataset
        self.merged_data = util.drop_column(self.merged_data, ["UOM","Cruise Distance","Distance"])
        return

    def drop_missing_rows (self, col_name:str)->None:
        """
        Remove rows with missing data in a specified column.

        This method removes rows from the DataFrame stored in the class instance
        where the specified column 'col_name' has missing (NaN) values. The DataFrame
        is modified in place.

        Parameters:
            self (DataPreprocessing): The instance of the class.
            col_name (str): The name of the column to check for missing data.

        Returns:
            None
        """
        self.merged_data.dropna(subset=[col_name], inplace=True)
        return

    def calculate_age_from_DOB(self, col_name:str)->None:
        """
        Calculate and add the age of individuals based on their date of birth.

        This method calculates the age of individuals using the 'Date of Birth' column
        specified in 'col_name' and adds a new 'Age' column to the DataFrame 'self.merged_data'.
        The calculated age is based on the current year.

        Parameters:
            self (DataPreprocessing): The instance of the class.
            col_name (str): The name of the 'Date of Birth' column.

        Returns:
            None
        """
        current_year = datetime.datetime.now().year
        self.merged_data["Age"] = current_year - self.merged_data[col_name].dt.year
        util.drop_column(self.merged_data, [col_name])
    
    def clean_and_prepare_data(self)-> None:
        """
        Clean and prepare the data by replacing specific values in columns.

        This method applies a series of value replacements in various columns of the DataFrame
        to clean and prepare the data for analysis. Specific replacements include renaming variations
        of "Blastoise" and "Lapras" in the "Cruise Name" column and removing specific values
        from other columns. The modifications are performed in place.

        Parameters:
            self (DataPreprocessing): The instance of the class.

        Returns:
            None
        """
        self.replace_values_in_column("Cruise Name",["blast", "blast0ise", "blastoise"],"Blastoise")
        self.replace_values_in_column("Cruise Name",["IAPRAS", "lap", "lapras"],"Lapras")
        self.replace_values_in_column("Embarkation/Disembarkation time convenient",[0],None)
        self.replace_values_in_column("Ease of Online booking",[0],None)
        self.replace_values_in_column("Online Check-in",[0],None)
        self.replace_values_in_column("Cabin Comfort",[0],None)
        self.replace_values_in_column("Onboard Service",[0],None)
        self.replace_values_in_column("Cleanliness",[0],None)
        self.replace_values_in_column("Embarkation/Disembarkation time convenient",[0],None)
        return
    
    def replace_values_in_column(self, col_name:str, replace_list:list[str], replace_with:str) -> None:
        """
        Replace specified values in a DataFrame column with a new value.

        This method iterates through the DataFrame column specified in 'col_name' and
        replaces each value from the 'replace_list' with the 'replace_with' value.
        The modified DataFrame is returned.

        Parameters:
            self (DataPreprocessing): The instance of the class.
            col_name (str): The name of the column to be modified.
            replace_list (list of str): A list of values to be replaced.
            replace_with (str): The value to replace the specified values with.

        Returns:
            None

        """
        for word in replace_list:
            self.merged_data.loc[self.merged_data[col_name]==word,col_name] = replace_with
        return 
