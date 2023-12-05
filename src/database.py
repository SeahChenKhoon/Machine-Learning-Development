import pandas as pd
import sqlite3
import util

def db_read(data_path, database:list)->pd.DataFrame:
    try:
        connection:connection = sqlite3.connect(data_path  + database['file_name'])
        dataframe:pd.DataFrame = pd.read_sql_query("SELECT * FROM " + database['table_name'], connection)
        return dataframe.set_index(database['index']).drop_duplicates()
    except sqlite3.Error as e:
        print("SQLite error:", e)
        return None

def db_merge_db(df_pre_cruise:pd.DataFrame, df_post_cruise:pd.DataFrame):
    return pd.merge(df_pre_cruise, df_post_cruise, left_index=True, right_index=True, how='inner')

