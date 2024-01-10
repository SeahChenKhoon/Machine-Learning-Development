import pandas as pd
import sqlite3
import util

def db_read(data_path: str, database:list)->pd.DataFrame:
    """
    Read data from an SQLite database and return it as a Pandas DataFrame.

    Parameters:
    - data_path (str): The path where the SQLite database file is located.
    - database (dict): A dictionary containing information about the database.
        - 'file_name' (str): The name of the SQLite database file.
        - 'table_name' (str): The name of the table to read from.
        - 'index' (str): The column to set as the DataFrame index.

    Returns:
    - pd.DataFrame or None: The DataFrame containing the data from the specified table, or None if an error occurs.

    Example: 
    - db_read("./data/", {'file_name': 'cruise_pre.db', 'table_name': 'cruise_pre', 'index': 'index'})
    """
    try:
        # Establish a connection to the SQLite database
        connection:connection = sqlite3.connect(data_path  + database['file_name'])

        # Read data from the specified table into a Pandas DataFrame
        dataframe:pd.DataFrame = pd.read_sql_query("SELECT * FROM " + database['table_name'], connection)

        # Set the specified column as the DataFrame index and drop duplicate rows
        return dataframe.set_index(database['index']).drop_duplicates()
    except sqlite3.Error as e:
        # Print an error message if there's an issue with the SQLite connection
        print("SQLite error:", e)
        return None

def db_merge_db(df_pre_cruise:pd.DataFrame, df_post_cruise:pd.DataFrame) -> pd.DataFrame:
    """
    Merge two Pandas DataFrames using their indices.

    Parameters:
    - df_pre_cruise (pd.DataFrame): The first DataFrame to be merged.
    - df_post_cruise (pd.DataFrame): The second DataFrame to be merged.

    Returns:
    - pd.DataFrame: The merged DataFrame based on the indices, using an inner join.

    Example:
    - db_merge_db (df_pre_cruise, df_post_cruise)
    """
    return pd.merge(df_pre_cruise, df_post_cruise, left_index=True, right_index=True, how='inner')

