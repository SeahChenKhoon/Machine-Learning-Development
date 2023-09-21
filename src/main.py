import data_preprocessing
import feature_engineering

datapath = "../data/"

# Perform Data Processing
#   1. Read source Data
#   2. Date of Birth - Remove Invalid datetime value  
#   3. Cruise Distance - Split col into  "UOM", "Distance in KM". Convert Miles to KM in Disances.
read_data = data_preprocessing.ReadData(datapath)
dataframe = read_data.read_data()
data_preprocessing = data_preprocessing.DataPreprocessing()
dataframe = data_preprocessing.process_data_preprocessing(dataframe)
fe = feature_engineering.Feature_Engineering()
dataframe = fe.impute_missing_data(dataframe)
dataframe = fe.convert_features_to_numeric(dataframe)

