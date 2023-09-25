import pandas as pd
import impute_missing_data
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
class ConvertToNumeric:
    def __init__(self, dataframe:pd.DataFrame)->None:
        self._dataframe = dataframe

class ConvertBinaryColumns(ConvertToNumeric):
    def process_conversion(self,col_name, binary_val:list):
        """
        Perform Binary encoding for the specific col_name in dataframe to set the first element to 0 and second to 1

        Parameters:
            col_name (str): The columns to be encoded.
            binary_val (list): A list contain 2 element to transform first element to 0 and second to 1
            
        Returns:
            dataframe (pd.DataFrame): Return back the processed dataset  
        """
        try:
            if len(binary_val) != 2:
                raise ValueError
            else:
                self._dataframe[col_name] = self._dataframe.apply(lambda row: 0 if row[col_name] == binary_val[0] else (1 if row[col_name] == binary_val[1] else row[col_name]), axis=1)
                return self._dataframe                
                dataframe = self._map_values_to_binary(dataframe,col_name,binary_val)
        except ValueError:
            print("This function expacts 2 elements in list.")
            exit()
        finally:
            return self._dataframe

class Ordinal_Encode(ConvertToNumeric):
    def process_conversion(self, list_column:list,impt_order:list):
        """
        Ordinally encodes a specified column in a DataFrame.

        Parameters:
            list_column (list): The name of the columns to encode.
            impt_order (list): A list of categories in the desired order.

        Returns:
            pd.DataFrame: The DataFrame with the specified column ordinally encoded.
        """
        for col_name in list_column:
            encoder = OrdinalEncoder(categories=[impt_order])
            self._dataframe[col_name] = encoder.fit_transform(self._dataframe[[col_name]])
            self._dataframe.loc[self._dataframe[col_name]==0,col_name] = None
        return self._dataframe
        

class LabelEncode(ConvertToNumeric):
    def process_conversion(self, col_name):
        """
        This function process the label encoding on the target column which is Ticket Type
        
        Parameters:
            col_name (str): Specify the column name within dataframe for the function perform processing.
            
        Returns:
            dataframe (pd.DataFrame): Return back the processed dataset
        """
        label_encoder = LabelEncoder()
        self._dataframe[col_name] = label_encoder.fit_transform(self._dataframe[col_name])
        return self._dataframe

class OneHotKeyEncode(ConvertToNumeric):
    def process_conversion(self, col_name):
        """
        This function process the one hot key encoding on the following column
        a> Source of Traffic

        Parameters:
            col_name ( list): Specify the column name within dataframe for the function perform processing.

        Returns:
            dataframe (pd.DataFrame): Return back the processed dataset    
        """
        encoded_df = pd.get_dummies(self._dataframe, columns=col_name)
        return encoded_df