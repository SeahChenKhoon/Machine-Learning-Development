import data_preprocessing
import feature_engineering
import util
import modelling
datapath = "./data/"

# Perform Data Processing
#   1. Read source Data
#   2. Date of Birth - Remove Invalid datetime value  
#   3. Cruise Distance - Split col into  "UOM", "Distance in KM". Convert Miles to KM in Disances.
read_data = data_preprocessing.ReadData(datapath)
dataframe = read_data.read_data()
data_preprocessing = data_preprocessing.DataPreprocessing()
dataframe = data_preprocessing.process_data_preprocessing(dataframe)
feature_engineering = feature_engineering.FeatureEngineer(dataframe)
dataframe = feature_engineering.fix_typo_error()
dataframe = feature_engineering.drop_ID_cols()
dataframe = feature_engineering.convert_features_to_numeric()
dataframe = feature_engineering.process_impute_missing_data()
util.output_csv(datapath,dataframe,"TheEnd")

random_forest_classifier = modelling.RandomForestClassify(dataframe,"Ticket Type",0.2)
model = random_forest_classifier.model()
accuracy, cv_score, model = random_forest_classifier.train_model(model)
print(f"accuracy: {accuracy}")
print(f"cv_score: {cv_score}")