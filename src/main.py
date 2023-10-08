import pandas as pd
import numpy as np
import warnings
import os

import data_preprocessing
import util

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay, \
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve

# Perform Data Processing
#   1. Read source Data
#   2. Date of Birth - Remove Invalid datetime value  
#   3. Cruise Distance - Split col into  "UOM", "Distance in KM". Convert Miles to KM in Disances.
warnings.filterwarnings('ignore')
os.environ['PYDEVD_DISABLE_FILE_VALIDATION'] = '1'

################################################################################
#                                      Preprocessing                           #
################################################################################
# Read Data
DATA_PATH = "./data/"
survey_scale= [None, 'Not at all important', 'A little important', 'Somewhat important',
            'Very important','Extremely important']
dp = data_preprocessing.DataPreprocessing(data_path=DATA_PATH,tablename_1="cruise_pre",tablename_2="cruise_post",index_col="index",
                                          survey_scale= [None, 'Not at all important', 'A little important', 'Somewhat important',
            'Very important','Extremely important'])
dp.load_data()
dp.merge_data()

###################
#  Data Cleansing #
###################
# Remove invalid date from DOB
dp.clean_datetime_col("Date of Birth")
# Convert "Cruise Distance" to "Distance in KM"
dp.mileage_conversion()
# Remove missing rows from target column
dp.drop_missing_rows("Ticket Type")
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
dp.label_encode(["Ticket Type"])
# Ordinal encode non-numeric ordinal variable
dp.ordinal_encode(["Onboard Wifi Service","Onboard Dining Service", "Onboard Entertainment"])     
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
dp.merged_data = util.drop_columns(dp.merged_data,["Ext_Intcode_x","Ext_Intcode_y","Logging","WiFi","Dining", "Entertainment"])

################################################################################
#                                      Preprocessing                           #
################################################################################
print(dp.merged_data.info())
util.output_csv(DATA_PATH,dp.merged_data,"TheEnd")

# x = dataframe.drop(["Ticket Type"], axis=1)
# y = dataframe["Ticket Type"]


#######
# PCA #
#######
# pca = PCA(n_components=i)
# X_pca = pca.fit_transform(x)



###################
# Standard Scaler #
###################
# std_scale = StandardScaler()
# std_scale.fit(x)
# dataframe = std_scale.transform(x)

###################
# Min-Max Scaler  #
###################
# minmax_scale = MinMaxScaler()
# minmax_scale.fit(x)
# x = minmax_scale.transform(x)


# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)

# base_classifier = DecisionTreeClassifier(max_depth=1)
# models={
#     # "Random Forest": RandomForestClassifier(),
#     # "SVM": SVC(decision_function_shape='ovr', kernel="sigmoid"), #0.3256
#     "XG Boost": xgb.XGBClassifier(
#             objective='multi:softmax',  # Multi-Class classification
#             verbosity=2
#             # max_depth=3,                   # Maximum tree depth
#             # learning_rate=0.1,             # Learning rate
#             # n_estimators=100,              # Number of boosting rounds (iterations)
#             # eval_metric='logloss'          # Evaluation metric (logarithmic loss)
#     ),
#     "Logistic Regression":LogisticRegression(multi_class='ovr')
#     # "Decision Tree": DecisionTreeClassifier(),
#     # "AdaBoost": AdaBoostClassifier(base_estimator=base_classifier,  # Weak classifier
#     #     n_estimators=50,                 # Number of weak classifiers to combine
#     #     learning_rate=1.0)
# }

# for i in range(len(list(models))):
#     model = list(models.values())[i]
#     model.fit(x_train,y_train)

#     y_train_pred = model.predict(x_train)
#     y_test_pred = model.predict(x_test)

#     # Training Set Performance
#     model_train_accuracy = accuracy_score(y_train, y_train_pred)
#     model_train_f1 = f1_score(y_train, y_train_pred, average="weighted")
#     model_train_precision = precision_score(y_train, y_train_pred, average="weighted")
#     model_train_recall = recall_score(y_train, y_train_pred, average="weighted")
#     # model_train_rocauc_score = roc_auc_score(y_train, y_train_pred, multi_class='ovo', needs_proba=True)

#     # Test Set Performance
#     model_test_accuracy = accuracy_score(y_test, y_test_pred)
#     model_test_f1 = f1_score(y_test, y_test_pred, average="weighted")
#     model_test_precision = precision_score(y_test, y_test_pred, average="weighted")
#     model_test_recall = recall_score(y_test, y_test_pred, average="weighted")
#     # model_test_rocauc_score = roc_auc_score(y_train, y_test_pred, multi_class='ovo', needs_proba=True)

#     print(list(models.keys())[i])

#     print("Model Performance for Training set")
#     print(" - Accuracy: {:.4f}".format(model_train_accuracy))
#     print(" - F1 score: {:.4f}".format(model_train_f1))
#     print(" - Precision: {:.4f}".format(model_train_precision))
#     print(" - Recall: {:.4f}".format(model_train_recall))
#     # print(" - Roc Auc Score: {:.4f}".format(model_train_rocauc_score))

#     print("--------------------------------------------------")
#     print("Model Performance for Test set")
#     print(" - Accuracy: {:.4f}".format(model_test_accuracy))
#     print(" - F1 score: {:.4f}".format(model_test_f1))
#     print(" - Precision: {:.4f}".format(model_test_precision))
#     print(" - Recall: {:.4f}".format(model_test_recall))
#     # print(" - Roc Auc Score: {:.4.f}".format(model_test_rocauc_score))
    
#     print('='*35)
#     print('\n')

# ##########################
# # HyperParameter Tuning  #
# ##########################
# # from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
# # rf_param = {"max_depth": [None,3,4,5,6,8, 10,12,15],
# #             "booster": ["gbtree"],
# #             "tree_method": ["auto","exact","approx","hist"],
# #             "alpha":[0,1,2,3,4,5],
# #             "lambda":[0,1,2,3,4,5],
# #             "colsample_bytree" : [0, 0.5, 1],
# #             "colsample_bylevel" : [0, 0.5, 1],
# #             "colsample_bynode" : [0, 0.5, 1],
# #             "grow_policy": ["depthwise","lossguide"],
# #             "learning_rate": [0.05, 0.10,0.15, 0.20, 0.25,0.30],
# #             "min_child_weight": [1,3,5,7],
# #             "gamma": [0.0,0.1,0.2,0.3,0.4]
# #             }

# # model_param = {}
# # random_search = RandomizedSearchCV(estimator=xgb.XGBClassifier(), param_distributions=rf_param,cv=5, 
# #                                    verbose=3,n_jobs=-1)
# # from datetime import datetime
# # def timer(start_time=None):
# #     if not start_time:
# #         start_time = datetime.now()
# #         return start_time
# #     elif start_time:
# #         thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
# #         tmin, tsec = divmod(temp_sec, 60)
# #         print("\n Time taken: %i hours %i minutes and %s seconds." % (thour, tmin, round(tsec,2)))
# # start_time = timer(None)
# # random_search.fit(x, y) 
# # start_time = timer(start_time)
# # print(random_search.best_estimator_)
# # print(random_search.best_params_)
# # best_estimator = random_search.best_estimator_

# ##########################
# # Final Tuning  #
# ##########################
# model = xgb.XGBClassifier(verbose=2, n_jobs=-1, booster= 'gbtree', 
#                           tree_method= 'auto', min_child_weight= 1, max_depth= 15, learning_rate= 0.25, 
#                           grow_policy= 'depthwise', gamma= 0.0, colsample_bytree= 1, colsample_bynode= 1, colsample_bylevel= 0.5, 
#                           alpha= 2)
# model.fit(x_train,y_train)

# y_train_pred = model.predict(x_train)
# y_test_pred = model.predict(x_test)

# # Training Set Performance
# model_train_accuracy = accuracy_score(y_train, y_train_pred)
# model_train_f1 = f1_score(y_train, y_train_pred, average="weighted")
# model_train_precision = precision_score(y_train, y_train_pred, average="weighted")
# model_train_recall = recall_score(y_train, y_train_pred, average="weighted")

# # Test Set Performance
# model_test_accuracy = accuracy_score(y_test, y_test_pred)
# model_test_f1 = f1_score(y_test, y_test_pred, average="weighted")
# model_test_precision = precision_score(y_test, y_test_pred, average="weighted")
# model_test_recall = recall_score(y_test, y_test_pred, average="weighted")

# print("XG Boost")
# print("Model Performance for Training set")
# print(" - Accuracy: {:.4f}".format(model_train_accuracy))
# print(" - F1 score: {:.4f}".format(model_train_f1))
# print(" - Precision: {:.4f}".format(model_train_precision))
# print(" - Recall: {:.4f}".format(model_train_recall))

# print("--------------------------------------------------")
# print("Model Performance for Test set")
# print(" - Accuracy: {:.4f}".format(model_test_accuracy))
# print(" - F1 score: {:.4f}".format(model_test_f1))
# print(" - Precision: {:.4f}".format(model_test_precision))
# print(" - Recall: {:.4f}".format(model_test_recall))



