import data_preprocessing
import test

datapath = "./data/"

# Perform Data Processing
#   1. Read source Data
#   2. Date of Birth - Remove Invalid datetime value  
#   3. Cruise Distance - Split col into  "UOM", "Distance in KM". Convert Miles to KM in Disances.
read_data = data_preprocessing.ReadData(datapath)
dataframe = read_data.read_data()
data_preprocessing = data_preprocessing.DataPreprocessing()
dataframe = data_preprocessing.process_data_preprocessing(dataframe)



# # Access the public method, which in turn calls the private method
# output = obj.public_method()

# # Print the result
# print(output)  # Output: The result is 10

# import util
# import model
# import feature_engineering
# data_preprocessing = DataPreprocessing.read_data()
# dataframe = data_preprocessing.read_data()
# print(dataframe.head(5))
# dataframe = feature_engineering.convert_columns_to_numeric(dataframe)
# dataframe = feature_engineering.impute_missing_data(dataframe)
# model.training_model(dataframe)
# util.output_csv(dataframe,"df_cruise_final")

