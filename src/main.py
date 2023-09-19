import data_preprocessing
import test

datapath = "./data/"

# Perform Data Processing
#   1. Read source Data
#   2. Cruise Name - Rectify typo error 
#   3. Date of Birth - Remove Invalid datetime value  
#   4. Cruise Distance - Split col into  "UOM", "Distance in KM". Convert Miles to KM in Disances.
dp = data_preprocessing.DataPreprocessing(datapath)
dataframe = dp.read_data()
dataframe = dp.fix_typo_error(dataframe,"Cruise Name",["blast", "blast0ise", "blastoise"],"Blastoise")
dataframe = dp.fix_typo_error(dataframe,"Cruise Name",["IAPRAS", "lap", "lapras"],"Lapras")
dataframe = dp.remove_invalid_data_in_datetime_col(dataframe,"Date of Birth")
dataframe = dp.convert_miles_to_km(dataframe,"Cruise Distance", ["Distance in KM", "UOM"])
print(dataframe[["Distance in KM"]].head())
print(dataframe[["Distance in KM"]].info())


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

