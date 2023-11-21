import sqlite3
import datetime
import util
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

class DataPreprocessing:
    def drop_column(self, columns_to_drop:list[str])->None:
        """
        Drop specified columns from the merged DataFrame.

        This method takes a list of column names and removes these columns from the merged DataFrame.
        
        Parameters:
            columns_to_drop (List[str]): A list of column names to be dropped from the DataFrame.

        Returns:
            None: The method updates the 'merged_data' attribute with the columns removed.
        """
        self.merged_data:pd.DataFrame = util.drop_columns(self.merged_data, columns_to_drop)
        return None

    def get_merged_data(self):
        return self.merged_data

    def get_x_y(self)->tuple[np.ndarray, pd.Series]:
        """
        Get the feature matrix (X) and target variable (y).

        This method returns the feature matrix (X) and the target variable (y) as a tuple.

        Returns:
            Tuple[np.ndarray, pd.Series]: A tuple containing the feature matrix (X) as a NumPy array and
            the target variable (y) as a Pandas Series.
        """
        return self.X, self.y

    def load_data(self, file_path:str, file_details:list)->None:
        """
        Load and merge data from multiple database files.

        This method loads data from a list of database files and merges it into a single DataFrame.
        
        Args:
            file_details (list of dict): A list of dictionaries, each containing information about a database file.
                Each dictionary should have the following keys:
                - 'file_path' (str): The file path to the database file.
                - 'table_name' (str): The name of the table in the database to read.
                - 'index' (str): The name of the index column to use for merging.

        Returns:
            None
        """
        first_file = True
        for datafile in file_details:
            dataframe:pd.DataFrame = self.read_db(file_path + datafile['file_name'], datafile['table_name'], datafile['index'])
            if first_file:
                self.merged_data = dataframe
                first_file = False
            else:
                self.merged_data = pd.merge(self.merged_data, dataframe, on=datafile['index'], how='inner')
        return None

    def read_db(self, file_path:str, table_name:str, index:str)->pd.DataFrame:
        """
        Read data from an SQLite database table.

        This method connects to an SQLite database at the specified file path, reads the data from a table,
        and returns it as a pandas DataFrame.

        Args:
            file_path (str): The file path to the SQLite database file.
            table_name (str): The name of the table in the database to read.
            index (str): The name of the index column to set as the DataFrame's index.

        Returns:
            pd.DataFrame: A pandas DataFrame containing the data from the specified table.

        """
        try:
            connection:connection = sqlite3.connect(file_path)
            dataframe:pd.DataFrame = pd.read_sql_query("SELECT * FROM " + table_name, connection)
            dataframe:pd.DataFrame = dataframe.set_index(index)
            return dataframe
        except sqlite3.Error as e:
            print("SQLite error:", e)
            return None

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
        return None

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
        temp_dist:pd.DataFrame = self.merged_data[(~self.merged_data["Distance"].isna())][["Distance","UOM"]]
        temp_dist["Distance in KM"] = pd.to_numeric(temp_dist['Distance'], errors='coerce')
        temp_dist.loc[temp_dist["UOM"]== "Miles","Distance in KM"] = temp_dist["Distance in KM"] * 1.60934
        temp_dist["Distance in KM"] = temp_dist["Distance in KM"].abs().round(0)
        temp_dist['Distance in KM'] = temp_dist['Distance in KM'].astype(int)

        self.merged_data.loc[~self.merged_data["Distance"].isnull(),"Distance in KM"] =  temp_dist["Distance in KM"]
        
        # 3> Drop "Cruise Distance" and "UOM" columns from dataset
        self.merged_data = util.drop_columns(self.merged_data, ["UOM","Cruise Distance","Distance"])
        return None

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
        return None

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
        return None
    
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
        label_encoder:LabelEncoder = LabelEncoder()
        for col_name in col_names:
            self.merged_data[col_name] = label_encoder.fit_transform(self.merged_data[col_name])
        return None

    def ordinal_encode(self, list_column:list[str], survey_scale:list[str]):
        """
        Perform ordinal encoding on specified columns in the merged DataFrame.

        This method performs ordinal encoding on the columns specified in `list_column` using the
        provided `survey_scale`. The ordinal encoding results are applied to the merged DataFrame.

        Parameters:
            list_column (list[str]): A list of column names to be ordinal encoded.
            survey_scale (list[str]): A list representing the ordinal scale for encoding.

        Returns:
            None: This method does not return a value.
        """
        encoder:OrdinalEncoder = OrdinalEncoder(categories=[survey_scale])
        for col_name in list_column:
            self.merged_data[col_name] = encoder.fit_transform(self.merged_data[[col_name]])
            self.merged_data.loc[self.merged_data[col_name]==0, col_name] = None
        return None

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
        return None

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
        return None 

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
        return None

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
        return None
    
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
        return None

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
        return None

    def remove_outlier(self, col_names:list[str])->None:
        """
        Perform removal of outliers record

        Parameters:
            col_names (list): Specify the column names within dataframe for checking and removal of outliers.

        Returns:
            dataframe (pd.DataFrame): Return back the processed dataset
        """
        for feature in col_names:
            Q1:float = self.merged_data[feature].quantile(0.25)
            Q3:float = self.merged_data[feature].quantile(0.75)
            IQR:float = Q3 - Q1
            lower_limit:float = Q1 - 1.5 * IQR
            upper_limit:float = Q3 + 1.5 * IQR
            self.merged_data = self.merged_data[(self.merged_data[feature] > lower_limit) & (self.merged_data[feature] < upper_limit)]
        return None

    def data_splitting(self):
        """
        Split the merged data into features (X) and the target variable (y).

        This method separates the merged data into two parts: the features (X) and the target variable (y).
        It drops the 'Ticket Type' column to create X and assigns the 'Ticket Type' column to y.

        Parameters:
            self (DataPreprocessing): The instance of the class.

        Returns:
            None: The method does not return a value, but it updates the 'X' and 'y' attributes of the class.
        """
        self.X:np.ndarray = self.merged_data.drop(["Ticket Type"], axis=1)
        self.y:pd.Series = self.merged_data["Ticket Type"]
        return None

    def standard_scaler (self)->None:
        """
        Standardize the features using StandardScaler.

        This method standardizes (scales) the feature data (self.X) using the StandardScaler
        from scikit-learn. It fits the scaler to the feature data and then transforms the
        feature data to have a mean of 0 and a standard deviation of 1.

        Parameters:
            self (DataPreprocessing): The instance of the class.

        Returns:
            None: The method does not return a value, but it updates the 'X' attribute of the class.
        """
        std_scale:StandardScaler = StandardScaler()
        std_scale.fit(self.X)
        self.X:np.ndarray = std_scale.transform(self.X)
        return None

    def min_max_scaler (self)->None:
        """
        Scale the features to a specified range using Min-Max scaling.

        This method scales (transforms) the feature data (self.X) using Min-Max scaling
        from scikit-learn. It fits the scaler to the feature data and then scales the
        feature data to a specified range, typically between 0 and 1.

        Parameters:
            self (DataPreprocessing): The instance of the class.

        Returns:
            None: The method does not return a value, but it updates the 'X' attribute of the class.
        """
        std_scale:MinMaxScaler = MinMaxScaler()
        std_scale.fit(self.X)
        self.X:np.ndarray = std_scale.transform(self.X)
        return None

    def train_test_split(self, test_size:float, random_state:int)->tuple[np.ndarray,np.ndarray,pd.Series, pd.Series]:
        """
        Split the dataset into training and testing sets.

        This method splits the feature matrix (X) and the target variable (y) into training and testing sets
        using the train_test_split function from scikit-learn.

        Parameters:
            test_size (float): The proportion of the dataset to include in the test split.
            random_state (int): The seed used by the random number generator for reproducibility.

        Returns:
            tuple: A tuple containing the following elements in this order:
                - X_train (np.ndarray): The feature matrix for the training set.
                - X_test (np.ndarray): The feature matrix for the testing set.
                - y_train (pd.Series): The target variable for the training set.
                - y_test (pd.Series): The target variable for the testing set.
        """
        return train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)
    
    def smote(self, test_size:float, random_state:int):
        """
        Applies Synthetic Minority Over-sampling Technique (SMOTE) to balance the dataset.

        Parameters:
        - test_size (float): The proportion of the synthetic samples to generate compared to the original dataset.
        - random_state (int): Seed for reproducibility.

        Returns:
        None: The method modifies the instance's X (features) and y (labels) attributes in-place.
        """
        smote = SMOTE(sampling_strategy='auto', random_state=random_state)
        self.X, self.y = smote.fit_resample(self.X, self.y)
        return
