import pandas as pd
import impute_missing_data
import util
import convert_features_to_numeric
import modelling

class FeatureEngineer:
    def __init__(self, dataframe:pd.DataFrame)->None:
        self._dataframe = dataframe

    def drop_ID_cols(self) -> pd.DataFrame:
        """
        Perform the dropping of ID columns
        """
        ID_cols = ["Logging", "Ext_Intcode","WiFi", "Dining","Entertainment"]
        self._dataframe = util.drop_column(self._dataframe,ID_cols)
        return self._dataframe

    def fix_typo_error(self)-> pd.DataFrame:
        """
        Perform fixing of typo error list by calling the _fix_typo_error

        Parameters:

        Returns:
            dataframe (pd.DataFrame): Return back the processed dataset 
        """
        self._dataframe = self._fix_typo_error("Cruise Name",["blast", "blast0ise", "blastoise"],"Blastoise")
        self._dataframe = self._fix_typo_error("Cruise Name",["IAPRAS", "lap", "lapras"],"Lapras")
        self._dataframe = self._fix_typo_error("Embarkation/Disembarkation time convenient",[0],None)
        self._dataframe = self._fix_typo_error("Ease of Online booking",[0],None)
        self._dataframe = self._fix_typo_error("Online Check-in",[0],None)
        self._dataframe = self._fix_typo_error("Cabin Comfort",[0],None)
        self._dataframe = self._fix_typo_error("Onboard Service",[0],None)
        self._dataframe = self._fix_typo_error("Cleanliness",[0],None)
        self._dataframe = self._fix_typo_error("Embarkation/Disembarkation time convenient",[0],None)
        return self._dataframe

    def _fix_typo_error(self, col_name:str, replace_list:list, replace_with:str) -> pd.DataFrame:
        """
        Perform fixing of typo error list of data (replace_list) to replace_with 

        Parameters:
            col_name (str): Specify the column name within dataframe for the function perform processing.
            replace_list (list): List of element to be replace
            replace_with (str): String to replace the element in replace_list

        Returns:
            dataframe (pd.DataFrame): Return back the processed dataset 
        """
        for word in replace_list:
            self._dataframe.loc[self._dataframe[col_name]==word,col_name] = replace_with
        return self._dataframe

    def process_impute_missing_data(self)->pd.DataFrame:
        impute_median = impute_missing_data.ImputeMedian(self._dataframe, ["Age"])
        self._dataframe = self._dataframe = impute_median.process_impute()
        impute_mean = impute_missing_data.ImputeMean(self._dataframe, ["Distance in KM"])
        self._dataframe = impute_mean.process_impute()
        impute_pred = impute_missing_data.ImputeUsingPred(self._dataframe,["Baggage handling", "Cabin service"])
        impute_pred.process_impute("Onboard Service")
        impute_pred = impute_missing_data.ImputeUsingPred(self._dataframe,["Cleanliness"])
        impute_pred.process_impute("Cabin service")
        impute_pred = impute_missing_data.ImputeUsingPred(self._dataframe,["Cabin service","Onboard Service"])
        impute_pred.process_impute("Baggage handling")

        impute_pred = impute_missing_data.ImputeUsingPred(self._dataframe,["Cabin Comfort","Online Check-in"])
        impute_pred.process_impute("Cleanliness")
        impute_pred = impute_missing_data.ImputeUsingPred(self._dataframe,["Cleanliness","Online Check-in"])
        impute_pred.process_impute("Cabin Comfort")
        impute_pred = impute_missing_data.ImputeUsingPred(self._dataframe,["Ease of Online booking","Cleanliness","Cabin Comfort"])
        impute_pred.process_impute("Online Check-in")
        impute_pred = impute_missing_data.ImputeUsingPred(self._dataframe,["Ease of Online booking","Gate location"])
        impute_pred.process_impute("Embarkation/Disembarkation time convenient")
        impute_pred = impute_missing_data.ImputeUsingPred(self._dataframe,["Embarkation/Disembarkation time convenient",
                                                                           "Gate location","Online Check-in"])
        impute_pred.process_impute("Ease of Online booking")
        impute_pred = impute_missing_data.ImputeUsingPred(self._dataframe,["Embarkation/Disembarkation time convenient",
                                                                           "Ease of Online booking"])
        impute_pred.process_impute("Gate location")
        impute_mode = impute_missing_data.ImputeMode(self._dataframe, ["Onboard Wifi Service", 
                                                                        "Embarkation/Disembarkation time convenient", 
                                                                        "Ease of Online booking","Gate location",
                                                                        "Onboard Dining Service", "Cabin Comfort",
                                                                        "Online Check-in","Onboard Entertainment","Cabin service",
                                                                        "Baggage handling","Port Check-in Service",
                                                                        "Onboard Service", "Cleanliness","Gender","Cruise Name"])
        self._dataframe = impute_mode.process_impute()
        return self._dataframe
    
    def convert_features_to_numeric(self)->pd.DataFrame:
        impt_order = [None, 'Not at all important', 'A little important', 'Somewhat important',
            'Very important','Extremely important']
        """
        Perform conversion of feature to numeric by
        1> 
        s>


		Parameters:			

		Returns:
			dataframe (pd.DataFrame): Return back the processed dataset
        """
        convert_binary_col = convert_features_to_numeric.ConvertBinaryColumns(self._dataframe)
        self._dataframe = convert_binary_col.process_conversion("Gender",["Female","Male"])
        self._dataframe = convert_binary_col.process_conversion("Cruise Name",["Blastoise","Lapras"])
        ordinal_encode = convert_features_to_numeric.Ordinal_Encode(self._dataframe)
        self._dataframe = ordinal_encode.process_conversion(["Onboard Wifi Service","Onboard Dining Service", "Onboard Entertainment"],
                                                            impt_order)        
        label_encode = convert_features_to_numeric.LabelEncode(self._dataframe)
        self._dataframe = label_encode.process_conversion("Ticket Type")
        ohk_encode = convert_features_to_numeric.OneHotKeyEncode(self._dataframe)
        self._dataframe = ohk_encode.process_conversion(["Source of Traffic"])
        return self._dataframe

