import pandas as pd
import sqlite3
import util

class Database():
    def __init__(self, data_path):
        self._data_path = data_path
        self._dataframe = pd.DataFrame()
        return None


    def db_read(self, database:list)->pd.DataFrame:
        try:
            connection:connection = sqlite3.connect(self._data_path   + database[]['file_name'])
            dataframe:pd.DataFrame = pd.read_sql_query("SELECT * FROM " + database['table_name'], connection)
            return dataframe.set_index(database['index']).drop_duplicates()
        except sqlite3.Error as e:
            print("SQLite error:", e)
            return None
        