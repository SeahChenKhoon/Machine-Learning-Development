import pandas as pd
import numpy as np
import warnings
import os

import data_preprocessing
import model_build
import util
import model_eval
import feature_engineering

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.model_selection import cross_validate


from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay, \
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve

warnings.filterwarnings('ignore')
os.environ['PYDEVD_DISABLE_FILE_VALIDATION'] = '1'

################################################################################
#                                      Preprocessing                           #
################################################################################
# Read Data
DATA_PATH:str = "./data/"
TARGET_VARIABLE:str = "Ticket Type"
survey_scale:list[str]= [None, 'Not at all important', 'A little important', 'Somewhat important',
            'Very important','Extremely important']
dp = data_preprocessing.DataPreprocessing()
data_file_1:dict={"filename":"cruise_pre","index":"index"}
data_file_2:dict={"filename":"cruise_post","index":"index"}
data_file_details:dict = {"datafile1":data_file_1, "datafile2":data_file_2}
dp.load_data(DATA_PATH, data_file_details)

config_file_path:str = 'config.yaml'
config:dict = util.read_config_file(config_file_path)

###################
#  Data Cleansing #
###################
# Remove invalid date from DOB
dp.clean_datetime_col("Date of Birth")
# Convert "Cruise Distance" to "Distance in KM"
dp.mileage_conversion()
# Remove missing rows from target column
dp.drop_missing_rows(TARGET_VARIABLE)
# Calculate age from DOB ****************** Should move to feature Engineering
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
dp.label_encode([TARGET_VARIABLE])
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
###########################
#  Remove Duplicated Rows #
###########################
dp.remove_duplicate_rows("Ext_Intcode_x")
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
################################################################################
#                             Feature Engineering                              #
################################################################################
fe = feature_engineering.FeatureEngineer()
fe.feature_grouping(dp.merged_data,["Onboard Wifi Service", "Onboard Dining Service", "Onboard Entertainment",
                     "Onboard Service"],"Onboard Survey")
fe.feature_grouping(dp.merged_data,["Embarkation/Disembarkation time convenient", "Ease of Online booking", 
                    "Gate location", "Online Check-in", "Port Check-in Service"],"Online Facility")
fe.feature_grouping(dp.merged_data,["Cabin Comfort", "Cabin service", "Baggage handling", 
                    "Cleanliness"],"Cabin Facility")
X:np.ndarray = None
y:pd.Series = None
X, y = dp.get_x_y()

x_train:np.ndarray = None
x_test:np.ndarray = None
y_train:pd.Series = None
y_test:pd.Series = None
x_train, x_test, y_train, y_test = dp.train_test_split(test_size=0.25,random_size=42)

########################
#  Logistic Regression #
########################
experimental_management:bool = None
if 'experimental_management' in config and 'enabled' in config['experimental_management'] \
  and config['experimental_management']['enabled']:
    experimental_management = True
else:
    experimental_management = False

model_type:str = "logistic_regression"
log_regress:model_build.LogRegression = model_build.LogRegression(x_train, x_test, y_train, y_test, experimental_management)
y_train_pred:np.ndarray = None
y_test_pred:np.ndarray = None
y_train_pred, y_test_pred = log_regress.modelling()
me:model_eval.ModelEval = model_eval.ModelEval(model_type, y_train, y_train_pred)
me.print_report("Training")

me = model_eval.ModelEval(model_type, y_test, y_test_pred)
me.print_report("Testing")

log_regress.process_hyperparameter(X, y)


##################
#  Random Forest #
##################
model_type:str = "random_forest"
random_forest:model_build.RandForest = model_build.RandForest(x_train, x_test, y_train, y_test, experimental_management)
y_train_pred:np.ndarray = None
y_test_pred:np.ndarray = None
y_train_pred, y_test_pred = random_forest.modelling()
me = model_eval.ModelEval(model_type, y_train, y_train_pred)
me.print_report("Training")

me = model_eval.ModelEval(model_type, y_test, y_test_pred)
me.print_report("Testing")

random_forest.process_hyperparameter(X, y)

#############
#  XG Boost #
#############
model_type = "XG Boost"
xg_boost:model_build.XgBoost = model_build.XgBoost(x_train, x_test, y_train, y_test, experimental_management)

y_train_pred:np.ndarray = None
y_test_pred:np.ndarray = None
y_train_pred, y_test_pred = xg_boost.modelling()
me = model_eval.ModelEval(model_type, y_train, y_train_pred)
me.print_report("Training")

me = model_eval.ModelEval(model_type, y_test, y_test_pred)
me.print_report("Testing")

random_forest.process_hyperparameter(X, y)

print(dp.merged_data.info())
util.output_csv(DATA_PATH,dp.merged_data,"TheEnd")


