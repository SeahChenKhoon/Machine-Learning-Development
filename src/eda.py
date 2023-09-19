    # * EDA (eda.py) which includes
    #     * Read Data. Read of 2 MD files into dataframes and combine them into one dataframes.
    #     * Standardise Cruise Distance to KM. This feature contains miles and km so standardise to km.
    #     * Convert DOB to Datetime by removing non-datetime records
    #     * Fix typo error in Cruise Name
import util
import data_mgmt
import pandas as pd
def eda()->pd.DataFrame:
    """
    Perform the following:
    * Read Data. Read of 2 MD files into dataframes and combine them into one dataframes.
    * Standardise Cruise Distance to KM. This feature contains miles and km so standardise to km.
    * Convert DOB to Datetime by removing non-datetime records
    * Fix typo error in Cruise Name

    Parameters:
        None
    Returns:
        dataframe (pd.DataFrame): Return back the processed dataset
    """

    dataframe = util.read_data()
    dataframe = data_mgmt.convert_miles_to_km(dataframe,"Cruise Distance")
    dataframe = util.convert_object_to_datetime(dataframe,'Date of Birth')
    dataframe = util.fix_typo_error(dataframe,"Cruise Name",["blast", "blast0ise", "blastoise"],"Blastoise")
    dataframe = util.fix_typo_error(dataframe,"Cruise Name",["IAPRAS", "lap", "lapras"],"Lapras")
    return dataframe