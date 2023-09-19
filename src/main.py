import data_preprocessing as dp
import test

datapath = "./data/"

# Create an instance of the dataprocessing to read cruise_pre.db and cruise_post.db
data_preprocessing = dp.DataPreprocessing(datapath)
dataframe = data_preprocessing.read_data()
# dataframe = data_preprocessing.fix_typo_error(dataframe,"Cruise Name",["blast", "blast0ise", "blastoise"],"Blastoise")
# dataframe = data_preprocessing.fix_typo_error(dataframe,"Cruise Name",["IAPRAS", "lap", "lapras"],"Lapras")



# print(dataframe["Cruise Name"].head(5))
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

