import pandas as pd
import impute_missing_data

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
        # impute_missing_data = impute_missing_data.ImputeMissingData(self._dataframe,["Age"],col_names)


# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import OrdinalEncoder
# from sklearn.preprocessing import LabelEncoder

# class Feature_Engineering:
#     def convert_features_to_numeric (self,dataframe: pd.DataFrame)->pd.DataFrame:
#         """
#         Perform feature_engineering on the following columns:
#             a> Convert columns with binary values to numeric 
#                 i> Gender - Female to 0 and Male to 1 
#                 ii> Cruise Name - Blastoise to 0 and Lapras to 1
#             b> Transform DOB to another new column Age
#             c> Perform Ordinal Encoding on 
#                 i> Onboard Wifi Service
#                 ii> Onboard Dining Service
#                 iii> Onboard Entertainment"
#             d> Perform Label Encode on "Ticket Type"
#             e> Perform One hot Key Encode on "Source of Traffic"
#         Parameters:
#             dataframe (pd.DataFrame): The dataset in which the feature enginneering will be acting on

#         Returns:
#             dataframe (pd.DataFrame): Return back the processed dataset
#         """
#         dataframe = self._fe_convert_binary_columns(dataframe,"Gender",["Female","Male"])
#         dataframe = self._fe_convert_binary_columns(dataframe,"Cruise Name",["Blastoise","Lapras"])
#         dataframe = self._fe_ordinal_encode(dataframe,["Onboard Wifi Service","Onboard Dining Service",
#             "Onboard Entertainment"])
#         dataframe = self._fe_label_encode_column(dataframe, "Ticket Type")
#         dataframe = self._fe_one_hot_key(dataframe, ["Source of Traffic"])
#         return dataframe

#     def _fe_convert_binary_columns(self, dataframe: pd.DataFrame, col_name:str, binary_val:list)-> pd.DataFrame:
#         """
#         Perform Binary encoding for the specific col_name in dataframe to set the first element to 0 and second to 1

#         Parameters:
#             dataframe (pd.DataFrame): The dataset in which the columns to be encoded
#             col_name (str): The columns to be encoded.
#             binary_val (list): A list contain 2 element to transform first element to 0 and second to 1
            
#         Returns:
#             dataframe (pd.DataFrame): Return back the processed dataset    
#         """
#         if len(binary_val) != 2:
#             print("This function expacts 2 elements in list.")
#             return dataframe
#         else:
#             dataframe = self._map_values_to_binary(dataframe,col_name,binary_val)
#         return dataframe

#     def _map_values_to_binary(self, dataframe:pd.DataFrame, col_name:str, data_val:list)->pd.DataFrame:
#         """
#         Map specific values in a DataFrame column to 0 and 1.

#         Parameters:
#             df (pd.DataFrame): The DataFrame to be modified.
#             col (str): The column name to map values in.
#             data_val (list): A list of two values to map to 0 and 1, respectively.

#         Returns:
#             pd.DataFrame: The modified DataFrame.
#         """

#         if len(data_val) != 2:
#             print("data_val should contain exactly two values.")
#             return dataframe

#         # dataframe[col_name] = dataframe[col_name].map({data_val[0]: 0, data_val[1]: 1})
#         dataframe[col_name] = dataframe.apply(lambda row: 0 if row[col_name] == data_val[0] else (1 if row[col_name] == data_val[1] else row[col_name]), axis=1)
#         return dataframe

#     def _fe_ordinal_encode(self, dataframe: pd.DataFrame, list_column:list)-> pd.DataFrame:
#         """
#         This function perform Ordinal Encode on the specified list of columns 
#             found in the specified dataframe
        
#         Parameters:
#             dataframe (pd.DataFrame): Specify dataset to be encode
#             list_column (list): Specify the list of importance
            
#         Returns:
#             dataframe (pd.DataFrame): Return back the processed dataset    
#         """
#         impt_order = [None, 'Not at all important', 'A little important', 'Somewhat important',
#             'Very important','Extremely important']
#         for col in list_column:
#             self._ordinal_encode(dataframe, col, impt_order)
#         return dataframe

#     def _ordinal_encode(self,dataframe, column_name, categories_order):
#         """
#         Ordinally encodes a specified column in a DataFrame.

#         Parameters:
#             dataframe (pd.DataFrame): The DataFrame containing the column to encode.
#             column_name (str): The name of the column to encode.
#             categories_order (list): A list of categories in the desired order.

#         Returns:
#             pd.DataFrame: The DataFrame with the specified column ordinally encoded.
#         """
#         encoder = OrdinalEncoder(categories=[categories_order])
#         dataframe[column_name] = encoder.fit_transform(dataframe[[column_name]])
#         dataframe.loc[dataframe[column_name]==0,column_name] = None
#         return dataframe

#     def _fe_label_encode_column(self,dataframe, col_name):
#         """
#         Perform label encoding on a specific column of a DataFrame.

#         Parameters:
#             dataframe (pd.DataFrame): The DataFrame to be encoded.
#             col_name (str): The name of the column to be label encoded.

#         Returns:
#             pd.DataFrame: The DataFrame with the specified column label encoded.
#         """
#         label_encoder = LabelEncoder()
#         dataframe[col_name] = label_encoder.fit_transform(dataframe[col_name])

#         return dataframe

#     def _fe_one_hot_key(self, dataframe: pd.DataFrame, col_name:list)-> pd.DataFrame:
#         """
#         This function process the one hot key encoding on the following column
#             a> Source of Traffic

#         Parameters:
#             col_name ( list) and dataframe (pd.DataFrame): Specify the column name within dataframe for the function perform processing.

#         Returns:
#             dataframe (pd.DataFrame): Return back the processed dataset    
#         """
#         dataframe =  self._fe_one_hot_encode(dataframe,col_name)
#         return dataframe    

#     def _fe_one_hot_encode(self, dataframe:pd.DataFrame, columns_to_encode:list)->pd.DataFrame:
#         """
#         Perform one-hot encoding on the specified columns of a DataFrame and concatenate the result.

#         Parameters:
#             df (pd.DataFrame): The DataFrame to be encoded.
#             columns_to_encode (list): A list of column names to be one-hot encoded.

#         Returns:
#             pd.DataFrame: The DataFrame with specified columns one-hot encoded and concatenated.
#         """
#         encoded_df = pd.get_dummies(dataframe, columns=columns_to_encode)
#         return encoded_df

#     def impute_missing_data(self,dataframe:pd.DataFrame):
#         dataframe= self._fe_impute_mode(dataframe,["Onboard Wifi Service","Embarkation/Disembarkation time convenient", 
#                                 "Ease of Online booking", "Gate location","Onboard Dining Service","Online Check-in","Cabin Comfort",
#                                 "Onboard Entertainment","Cabin service","Baggage handling","Port Check-in Service","Onboard Service",
#                                 "Cleanliness","Cruise Name"])
#         dataframe=self._fe_impute_median(dataframe,["Age"])
#         dataframe=self._fe_impute_mean(dataframe,["Distance in KM"])
#         dataframe=self._fe_impute_random(dataframe, ["Gender"])
#         return dataframe

#     def _fe_impute_random(self,dataframe:pd.DataFrame, col_names:list)->pd.DataFrame:
#         """
#         Interact with util.impute_rand to 
#         1> Replace the missing value with its random
#         2> Create a new colume *_rand to indicate there was a missing value

#         Parameters:
#             col_name (list) and dataframe (pd.DataFrame): Specify the column names within dataframe for the function perform processing of random.
            
#         Returns:
#             dataframe (pd.DataFrame): Return back the processed dataset    
#         """ 
#         for col_name in col_names:
#             dataframe = self._impute_random(dataframe,col_name)
#         return dataframe

#     def _impute_random(self, dataframe, col_name, add_mark_col=False)->pd.DataFrame:
#         """
#         Perform the following based on the specified column name and dataset 
#         1> the creation of new column col_name + "_random" to track missing value indication
#         2> Impute the missing value with its random value within the existing value range

#         Parameters:
#             col_name (str) and dataframe (pd.DataFrame): Specify the column name within dataframe for the function perform processing.
            

#         Returns:
#             dataframe (pd.DataFrame): Return back the processed dataset
#         """
#         dataframe[col_name + "_rand"] = np.where(dataframe[col_name].isnull(),1,0)
#         random_values = dataframe[col_name].dropna().sample(dataframe[col_name].isna().sum(), replace=True).values
#         dataframe.loc[dataframe[col_name].isna(), col_name] = random_values
#         return dataframe

#     def _fe_impute_mean(self, dataframe:pd.DataFrame, col_names:list)->pd.DataFrame:
#         """
#         Interact with util.impute_mode to 
#         1> Replace the missing value with its mean
#         2> Create a new colume *_mean to indicate there was a missing value

#         Parameters:
#             col_name (list) and dataframe (pd.DataFrame): Specify the column names within dataframe for the function perform processing of mean.
            
#         Returns:
#             dataframe (pd.DataFrame): Return back the processed dataset    
#         """ 
#         for col_name in col_names:
#             dataframe = self._impute_mean(dataframe,col_name)
#         return dataframe

#     def _impute_mean(self,dataframe:pd.DataFrame, col_name, add_mark_col=False)->pd.DataFrame:
#         """
#         Perform the following based on the specified column name and dataset 
#         1> the creation of new column col_name + "_mean" to track missing value indication
#         2> Impute the missing value with its mean value 

#         Parameters:
#             col_name (str) and dataframe (pd.DataFrame): Specify the column name within dataframe for the function perform processing.
            

#         Returns:
#             dataframe (pd.DataFrame): Return back the processed dataset
#         """
#         if add_mark_col:
#             dataframe[col_name + "_mean"] = np.where(dataframe[col_name].isnull(),1,0)
#         dataframe[col_name] = dataframe[col_name].fillna(int(dataframe[col_name].mean()))
#         return dataframe

#     def _fe_impute_median(self,dataframe:pd.DataFrame, col_names:list)->pd.DataFrame:
#         """
#         Interact with util.impute_mode to 
#         1> Replace the missing value with its median
#         2> Create a new colume *_median to indicate there was a missing value

#         Parameters:
#             col_name (list) and dataframe (pd.DataFrame): Specify the column names within dataframe for the function perform processing of median.
            
#         Returns:
#             dataframe (pd.DataFrame): Return back the processed dataset    
#         """ 
#         for col_name in col_names:
#             dataframe = self.impute_mean(dataframe,col_name)
#         return dataframe

#     def _fe_impute_median(self,dataframe:pd.DataFrame, col_name, add_mark_col=False)->pd.DataFrame:
#         """
#         Perform Imputation of specific column in dataset to median

# 		Parameters:
# 			col_name (str) and dataframe (pd.DataFrame): Specify the column name within dataframe for the function perform processing.
			

# 		Returns:
# 			dataframe (pd.DataFrame): Return back the processed dataset
#         """
#         if add_mark_col:
#             dataframe[col_name + "_median"] = np.where(dataframe[col_name].isnull(),1,0)
#         dataframe[col_name] = dataframe[col_name].fillna(int(dataframe[col_name].median()))
#         return dataframe


#     def _fe_impute_mode(self,dataframe:pd.DataFrame, col_names:list)->pd.DataFrame:
#         """
#         Pass each of the specific column in the list for imputation of mode
        
#         Parameters:
#             col_name (str) and dataframe (pd.DataFrame): Specify the column name within dataframe for the function perform processing.
            
#         Returns:
#             dataframe (pd.DataFrame): Return back the processed dataset
#         """
#         for col_name in col_names:
#             dataframe = self._impute_mode(dataframe,col_name)
#         return dataframe
    
#     def _impute_mode(self, dataframe:pd.DataFrame, col_name:str, add_mark_col=False)->pd.DataFrame:
#         """
#         Perform the following based on the specified column name and dataset 
#         1> the creation of new column col_name + "_mode" to track missing value indication
#         2> Impute the missing value with its mode

#         Parameters:
#             col_name (str) and dataframe (pd.DataFrame): Specify the column name within dataframe for the function perform processing.
            
#         Returns:
#             dataframe (pd.DataFrame): Return back the processed dataset
#         """
#         if add_mark_col:
#             dataframe[col_name + "_mode"] = np.where(dataframe[col_name].isnull(),1,0)
#         dataframe[col_name] = dataframe[col_name].fillna(dataframe[col_name].mode()[0])
#         return dataframe
    