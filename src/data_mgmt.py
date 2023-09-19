import util
import pandas as pd

def convert_miles_to_km(dataframe:pd.DataFrame, col_name:str)->None:
    """
        This function performs the following using the column_name and dataframe:
        1> Strip the distance away from the unit into 2 new columns "Distance" and "UOM". 
        2> Convert the distance into KM into "Distance in KM"
        3> Drop all the working columns with the input column
    Parameters:
        col_name (str) and dataframe (pd.DataFrame): 
        Specify the column name within dataframe for the function perform processing.
        
    Returns:
        dataframe (pd.DataFrame): 
    """
    # 1> Strip the distance away from the unit into 2 new columns "Distance" and "UOM"
    dataframe[["Distance", "UOM"]] = dataframe["Cruise Distance"].str.split(pat=' ', n=1, expand=True)
    convert_text_to_numeric_col(dataframe,'Distance')
    # 2> Convert the distance into KM.
    conversion_factors = {'Miles': 1.60934, 'KM': 1.0}
    dataframe['Distance in KM'] = abs(round(dataframe['Distance'] * dataframe['UOM'].map(conversion_factors),0))
    dataframe.drop("Distance",axis=1,inplace=True)
    dataframe.drop("UOM",axis=1,inplace=True)
    dataframe.drop("Cruise Distance",axis=1,inplace=True)    
    return dataframe

def convert_text_to_numeric_col (dataframe: pd.DataFrame, feature: str)->pd.Series:
    """
    Summary:
        This function convert the specific variable column from Object to Numeric Data Type
    """
    dataframe[feature] = pd.to_numeric(dataframe[feature], errors='coerce')


