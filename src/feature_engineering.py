import pandas as pd
import impute_missing_data
import convert_features_to_numeric

class FeatureEngineer:
    def __init__(self, dataframe:pd.DataFrame)->None:
        self._dataframe = dataframe

    def process_impute_missing_data(self)->pd.DataFrame:
        impute_median = impute_missing_data.ImputeMedian(self._dataframe, ["Age"])
        self._dataframe = self._dataframe = impute_median.process_impute()
        impute_mean = impute_missing_data.ImputeMean(self._dataframe, ["Distance in KM"])
        self._dataframe = impute_mean.process_impute()
        impute_mode = impute_missing_data.ImputeMode(self._dataframe, ["Age"])
        self._dataframe = impute_mode.process_impute()
        impute_random = impute_missing_data.ImputeRandom(self._dataframe, ["Gender"])
        self._dataframe = impute_random.process_impute()
        return self._dataframe
    
    def convert_features_to_numeric(self)->pd.DataFrame:
        impt_order = [None, 'Not at all important', 'A little important', 'Somewhat important',
            'Very important','Extremely important']
        """
        dataframe = self._fe_convert_binary_columns(dataframe,"Gender",["Female","Male"])
        dataframe = self._fe_convert_binary_columns(dataframe,"Cruise Name",["Blastoise","Lapras"])
        dataframe = self._fe_ordinal_encode(dataframe,["Onboard Wifi Service","Onboard Dining Service",
            "Onboard Entertainment"])
        dataframe = self._fe_label_encode_column(dataframe, "Ticket Type")
        dataframe = self._fe_one_hot_key(dataframe, ["Source of Traffic"])
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

