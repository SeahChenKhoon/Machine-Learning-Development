import sqlite3
import datetime
import util
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder


class DataPreprocessing:
    def __init__(self, data_path:str, tablename_1:str, tablename_2:str, index_col:str, survey_scale:list[str])->None:
        """
        Initialize a DataPreprocessing object.

        This constructor sets up the DataPreprocessing object with the necessary
        information to load and process data from two data files.

        Parameters:
            self (DataPreprocessing): : The instance of the class.
            data_path (str): The base path where the data files are located.
            tablename_1 (str): The name of the first table name to be processed.
            tablename_2 (str): The name of the second table name to be processed.
            index_col (str): The common column that joins the 2 data files.
            survey_scale (list[str]): The scale matrix used in the survey.

        Returns:
            None
        """
        self.data_path:str =  data_path
        self.tablename_1:str = tablename_1
        self.tablename_2:str = tablename_2
        self.index_col:str = index_col
        self.survey_scale = survey_scale
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
        self.merged_data = util.drop_columns(self.merged_data, ["UOM","Cruise Distance","Distance"])
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
        util.drop_columns(self.merged_data, [col_name])
    
    def label_encode(self, col_names:list[str])->None:
        """
        Encode the specified columns in the DataFrame using Label Encoding.

        Label Encoding is used to convert categorical values in the specified columns into
        numeric values. Each unique category in a column is assigned a unique integer label.

        Parameters:
        - self (DataPreprocessing): The instance of the class.
        - col_names (list of str): A list of column names to be label encoded.

        Returns:
        - None
        """
        label_encoder = LabelEncoder()
        for col_name in col_names:
            self.merged_data[col_name] = label_encoder.fit_transform(self.merged_data[col_name])

    def ordinal_encode(self, list_column:list[str]):
        """
        Perform ordinal encoding and handle zero values in specified columns of the DataFrame.

        Ordinal Encoding is used to convert categorical values in the specified columns into
        numeric values while preserving their ordinal relationships. Zero values are treated as
        missing values and replaced with 'None'.

        Parameters:
        self (DataPreprocessing): The instance of the class.
        list_column (list of str): A list of column names to be processed.

        Returns:
        - None
        """
        encoder = OrdinalEncoder(categories=[self.survey_scale])
        for col_name in list_column:
            self.merged_data[col_name] = encoder.fit_transform(self.merged_data[[col_name]])
            self.merged_data.loc[self.merged_data[col_name]==0, col_name] = None

    def one_hot_key_encode(self, col_names:list[str]):
        """
        Perform one-hot encoding on specified columns in the DataFrame.

        One-Hot Encoding is used to convert categorical values in the specified columns into
        binary columns, where each unique category becomes a new binary column (0 or 1).
        The original columns are dropped to avoid multicollinearity.

        Parameters:
        - col_names (list of str): A list of column names to be one-hot encoded.

        Returns:
        - None
        """
        self.merged_data = pd.get_dummies(self.merged_data, columns=col_names, drop_first=True)

    # def convert_features_to_numeric(self)->pd.DataFrame:
        # impt_order = [None, 'Not at all important', 'A little important', 'Somewhat important',
        #     'Very important','Extremely important']
        # convert_binary_col = self.ConvertBinaryColumns(self._dataframe)
        # self._dataframe = convert_binary_col.process_conversion("Gender",["Female","Male"])
        # self._dataframe = convert_binary_col.process_conversion("Cruise Name",["Blastoise","Lapras"])
        # ordinal_encode = convert_features_to_numeric.Ordinal_Encode(self._dataframe)
        # self._dataframe = ordinal_encode.process_conversion(["Onboard Wifi Service","Onboard Dining Service", "Onboard Entertainment"],
        #                                                     impt_order)        
        # label_encode = convert_features_to_numeric.LabelEncode(self._dataframe)
        # self._dataframe = label_encode.process_conversion("Ticket Type")
        # ohk_encode = convert_features_to_numeric.OneHotKeyEncode(self._dataframe)
        # self._dataframe = ohk_encode.process_conversion(["Source of Traffic"])
        # return self._dataframe

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

    def impute_median(self, col_names:list[str])->None:
        """
        Impute missing values in specified columns with the median value.

        Missing values in the specified columns are replaced with the median value of each
        respective column.

        Parameters:
        - col_names (list of str): A list of column names to be imputed with the median.

        Returns:
        - None
        """
        for col_name in col_names:
            self.merged_data[col_name].fillna(int(self.merged_data[col_name].median()),inplace=True)    

    def impute_mean(self, col_names:list[str])->None:
        """
        Impute missing values in specified columns with the mean value.

        Missing values in the specified columns are replaced with the mean value of each
        respective column.

        Parameters:
        - col_names (list of str): A list of column names to be imputed with the mean.

        Returns:
        - None
        """
        for col_name in col_names:
            self.merged_data[col_name].fillna(int(self.merged_data[col_name].mean()), inplace=True)
    
    def impute_mode(self, col_names:list[str])->None:
        """
        Impute missing values in specified columns with the mode value.

        Missing values in the specified columns are replaced with the mode (most frequent
        value) of each respective column.

        Parameters:
        - col_names (list of str): A list of column names to be imputed with the mode.

        Returns:
        - None
        """
        for col_name in col_names:
            mode = self.merged_data[col_name].mode()
            self.merged_data[col_name].fillna(int(mode.iloc[0]), inplace=True)

    def remove_duplicate_rows(self, col_name:str) -> None:
        """
        Remove duplicate rows from the DataFrame based on a specified column.

        This function identifies and removes duplicate rows in the DataFrame based on the values
        in the specified column, keeping the last occurrence of each duplicated value.

        Parameters:
        - col_name (str): The name of the column used to identify duplicate rows.

        Returns:
        - None
        """
        self.merged_data.drop_duplicates(subset=[col_name], keep="last", inplace=True)
        # self._dataframe.reset_index(inplace=True)
        # all_dup_idx = set(self._dataframe["index"].loc[self._dataframe.duplicated(subset=["Ext_Intcode_x"], keep=False)])
        # last_dup_idx = set(self._dataframe["index"].loc[self._dataframe.duplicated(subset=["Ext_Intcode_x"], keep="first")])
        # dup_idx_to_remove = all_dup_idx.symmetric_difference(last_dup_idx)
        # return self._dataframe

    def remove_outlier(self, col_names:list[str])->None:
        """
        Perform removal of outliers record

        Parameters:
            col_names (list): Specify the column names within dataframe for checking and removal of outliers.

        Returns:
            dataframe (pd.DataFrame): Return back the processed dataset
        """
        for feature in col_names:
            Q1 = self.merged_data[feature].quantile(0.25)
            Q3 = self.merged_data[feature].quantile(0.75)
            IQR = Q3 - Q1
            lower_limit = Q1 - 1.5 * IQR
            upper_limit = Q3 + 1.5 * IQR
            self.merged_data = self.merged_data[(self.merged_data[feature] > lower_limit) & (self.merged_data[feature] < upper_limit)]
