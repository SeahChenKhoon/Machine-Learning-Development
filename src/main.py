import eda
import util
import model
import feature_engineering


dataframe = eda.eda()
dataframe = feature_engineering.convert_columns_to_numeric(dataframe)
dataframe = feature_engineering.impute_missing_data(dataframe)
model.training_model(dataframe)
util.output_csv(dataframe,"df_cruise_final")

