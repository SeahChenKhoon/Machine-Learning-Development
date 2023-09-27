import data_preprocessing
import feature_engineering
import util
import numpy as np
import modelling
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
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

# random_forest_classifier = modelling.RandomForestClassify(dataframe,"Ticket Type",0.2)
# model = random_forest_classifier.model()
# accuracy, cv_score, model = random_forest_classifier.train_model(model)
# print(f"accuracy: {accuracy}")
# print(f"cv_score: {cv_score}")
x = dataframe.drop(["Ticket Type"], axis=1)
y = dataframe["Ticket Type"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

# minmax_scale = MinMaxScaler()
# minmax_scale.fit(x_train)
# x_train = minmax_scale.transform(x_train)

# std_scale = StandardScaler()
# std_scale.fit(x_train)
# x_train = std_scale.transform(x_train)

logistic_regression = LogisticRegression(max_iter=5000)
logistic_regression.fit(x_train,y_train)

accuracy = logistic_regression.score(x_train, y_train)
print(f"accuracy: {accuracy}")
# score = cross_val_score(logistic_regression, x, y, cv=5)
# cv_score = np.mean(score)
# print(f"cv_score: {cv_score}")

