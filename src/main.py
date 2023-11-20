import pandas as pd
import numpy as np
import warnings
import os
import yaml

import data_preprocessing
import model_build
import util
import model_eval
import feature_engineering
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay, \
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve

warnings.filterwarnings('ignore')
os.environ['PYDEVD_DISABLE_FILE_VALIDATION'] = '1'

################################################################################
#                                      Preprocessing                           #
################################################################################
survey_scale:list[str]= [None, 'Not at all important', 'A little important', 'Somewhat important',
            'Very important','Extremely important']

# Read config in yaml file
with open('../config.yaml', 'r') as yaml_file:
    data = yaml.safe_load(yaml_file)

dp = data_preprocessing.DataPreprocessing()
dp.load_data(data['databases'])

target_variable = data['target_variable']

###################
#  Data Cleansing #
###################
#  Remove Duplicated Rows 
dp.remove_duplicate_rows("Ext_Intcode_x")
# Remove invalid date from DOB
dp.clean_datetime_col("Date of Birth")
# Convert "Cruise Distance" to "Distance in KM"
dp.mileage_conversion()
# Remove missing rows from target column
dp.drop_missing_rows(target_variable)
# Calculate age from DOB
dp.calculate_age_from_DOB("Date of Birth")

# Data Cleansing invalid values
dp.replace_values_in_column("Cruise Name",["blast", "blast0ise", "blastoise"],"Blastoise")
dp.replace_values_in_column("Cruise Name",["IAPRAS", "lap", "lapras"],"Lapras")
dp.replace_values_in_column("Embarkation/Disembarkation time convenient",[0],None)
dp.replace_values_in_column("Ease of Online booking",[0],None)
dp.replace_values_in_column("Online Check-in",[0],None)
dp.replace_values_in_column("Cabin Comfort",[0],None)
dp.replace_values_in_column("Onboard Service",[0],None)
dp.replace_values_in_column("Cleanliness",[0],None)
dp.replace_values_in_column("Embarkation/Disembarkation time convenient",[0],None)

# Convert binary categorical values to numeric values
dp.label_encode(["Gender", "Cruise Name"])
# Label encode target variable
dp.label_encode([target_variable])
# Ordinal encode non-numeric ordinal variable
dp.ordinal_encode(["Onboard Wifi Service","Onboard Dining Service", "Onboard Entertainment"], survey_scale)     
# Perform one hot key encode on Source of Traffic
dp.one_hot_key_encode(["Source of Traffic"])

########################
#  Impute Missing Data #
########################
dp.impute_median(["Age"])
dp.impute_mean(["Distance in KM"])
dp.impute_mode(["Onboard Wifi Service", "Embarkation/Disembarkation time convenient", 
                "Ease of Online booking","Gate location", "Onboard Dining Service", 
                "Cabin Comfort", "Online Check-in","Onboard Entertainment","Cabin service",
                "Baggage handling", "Port Check-in Service", "Onboard Service", 
                "Cleanliness", "Gender", "Cruise Name", "WiFi", "Dining","Entertainment"])
####################
#  Remove outliers #
####################
dp.remove_outlier(["Age","Distance in KM"])
############################################
#  Drop ID, and irrelvant post-cruise data #
############################################
dp.drop_column(["Ext_Intcode_x", "Ext_Intcode_y", "Logging", "WiFi", "Dining", "Entertainment"])
dp.data_splitting()
dp.standard_scaler()

# ################################################################################
# #                             Feature Engineering                              #
# ################################################################################
# fe = feature_engineering.FeatureEngineer()
# fe.feature_grouping(dp.merged_data,["Ease of Online booking", "Online Check-in", "Port Check-in Service"],
#                     "Booking and Check-In")
# fe.feature_grouping(dp.merged_data,["Onboard Wifi Service", "Onboard Dining Service", 
#                     "Onboard Entertainment", "Onboard Service"],"Onboard Services")
# fe.feature_grouping(dp.merged_data,["Cabin Comfort", "Cabin service"],"Cabin and Comfort")

X:np.ndarray = None
y:pd.Series = None
X, y = dp.get_x_y()

x_train:np.ndarray = None
x_test:np.ndarray = None
y_train:pd.Series = None
y_test:pd.Series = None
x_train, x_test, y_train, y_test = dp.train_test_split(test_size=0.25,random_size=42)

##############
#  Modelling #
##############
for model in data['selected_model']:
    model_name = model['model_name']
    hyperparameter_enabled = model['hyperparameter_enabled']
    hyperparameter_tuning = model['hyperparameter_tuning']
    if model_name == "logistic_regression":
        model:model_build.LogRegression = model_build.LogRegression(model_name,x_train, x_test, y_train, y_test, 
            hyperparameter_enabled, hyperparameter_tuning)
    elif model_name == "random_forest":
        model:model_build.RandForest = model_build.RandForest(model_name,x_train, x_test, y_train, y_test, 
            hyperparameter_enabled, hyperparameter_tuning)
    elif model_name == "XG Boost":
        model:model_build.XgBoost = model_build.XgBoost(model_name, x_train, x_test, y_train, y_test, 
            hyperparameter_enabled, hyperparameter_tuning)
    else:
        print(f"{model_name} not found!")
        continue
    y_train_pred:np.ndarray = None
    y_test_pred:np.ndarray = None
    y_train_pred, y_test_pred = model.modelling()
    me:model_eval.ModelEval = model_eval.ModelEval(model_name, y_train, y_train_pred)
    me.print_report("Training")

    me = model_eval.ModelEval(model_name, y_test, y_test_pred)
    me.print_report("Testing")

    model.process_hyperparameter(X, y)

