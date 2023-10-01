import data_preprocessing
import feature_engineering
import util
import pandas as pd
import numpy as np
import modelling
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay, \
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve
datapath = "./data/"



# Perform Data Processing
#   1. Read source Data
#   2. Date of Birth - Remove Invalid datetime value  
#   3. Cruise Distance - Split col into  "UOM", "Distance in KM". Convert Miles to KM in Disances.
warnings.filterwarnings('ignore')
read_data = data_preprocessing.ReadData(datapath)
dataframe = read_data.read_data()
data_preprocessing = data_preprocessing.DataPreprocessing()
dataframe = data_preprocessing.process_data_preprocessing(dataframe)
feature_engineering = feature_engineering.FeatureEngineer(dataframe)
# dataframe = feature_engineering.drop_duplicate_rows()
dataframe = feature_engineering.fix_typo_error()
dataframe = feature_engineering.convert_features_to_numeric()
dataframe = feature_engineering.process_impute_missing_data()
# dataframe = feature_engineering.remove_outlier()
dataframe = feature_engineering.drop_ID_cols()
util.output_csv(datapath,dataframe,"TheEnd")

x = dataframe.drop(["Ticket Type"], axis=1)
y = dataframe["Ticket Type"]

#######
# PCA #
#######
# pca = PCA(n_components=i)
# X_pca = pca.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)

###################
# Standard Scaler #
###################
# std_scale = StandardScaler()
# std_scale.fit(x_train)
# x_train = std_scale.transform(x_train)

###################
# Min-Max Scaler  #
###################
# minmax_scale = MinMaxScaler()
# minmax_scale.fit(x_train)
# x_train = minmax_scale.transform(x_train)


models={
    "Random Forest": RandomForestClassifier(),
    "Logistic Regression":LogisticRegression(max_iter=5000),
    "Decision Tree": DecisionTreeClassifier()
}

for i in range(len(list(models))):
    model = list(models.values())[i]
    model.fit(x_train,y_train)

    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    # Training Set Performance
    model_train_accuracy = accuracy_score(y_train, y_train_pred)
    model_train_f1 = f1_score(y_train, y_train_pred, average="weighted")
    model_train_precision = precision_score(y_train, y_train_pred, average="weighted")
    model_train_recall = recall_score(y_train, y_train_pred, average="weighted")
    # model_train_rocauc_score = roc_auc_score(y_train, y_train_pred, multi_class='ovo', needs_proba=True)

    # Test Set Performance
    model_test_accuracy = accuracy_score(y_test, y_test_pred)
    model_test_f1 = f1_score(y_test, y_test_pred, average="weighted")
    model_test_precision = precision_score(y_test, y_test_pred, average="weighted")
    model_test_recall = recall_score(y_test, y_test_pred, average="weighted")
    # model_test_rocauc_score = roc_auc_score(y_train, y_test_pred, multi_class='ovo', needs_proba=True)

    print(list(models.keys())[i])

    print("Model Performance for Training set")
    print(" - Accuracy: {:.4f}".format(model_train_accuracy))
    print(" - F1 score: {:.4f}".format(model_train_f1))
    print(" - Precision: {:.4f}".format(model_train_precision))
    print(" - Recall: {:.4f}".format(model_train_recall))
    # print(" - Roc Auc Score: {:.4f}".format(model_train_rocauc_score))

    print("--------------------------------------------------")
    print("Model Performance for Test set")
    print(" - Accuracy: {:.4f}".format(model_test_accuracy))
    print(" - F1 score: {:.4f}".format(model_test_f1))
    print(" - Precision: {:.4f}".format(model_test_precision))
    print(" - Recall: {:.4f}".format(model_test_recall))
    # print(" - Roc Auc Score: {:.4.f}".format(model_test_rocauc_score))
    
    print('='*35)
    print('\n')

# ##########################
# # HyperParameter Tuning  #
# ##########################
# from sklearn.model_selection import GridSearchCV
# # HyperParameter Training
# rf_param = {"max_depth": [None,5, 8, 10, 15],
#             "max_features": [5,7, 8, "auto"],
#             "min_samples_split": [2, 8, 15,20],
#             "n_estimators": [100, 200, 500, 1000]}
# # rf_param = {"max_depth": [None,5],
# #             "max_features": [5,7]}
# model_param = {}
# grid_search_cv = GridSearchCV(estimator=RandomForestClassifier(), param_grid=rf_param, cv=5, verbose=2,n_jobs=-1)
# grid_search_cv.fit(x, y) 

# print(grid_search_cv.cv_results_)
# df_cv_result = pd.DataFrame(grid_search_cv.cv_results_)
# df_cv_result = df_cv_result[["params", "mean_test_score","mean_fit_time"]]
# df_cv_result = pd.concat([df_cv_result.drop(['params'], axis=1), df_cv_result['params'].apply(pd.Series)], axis=1)
# util.output_csv(datapath,df_cv_result,"CV_result")
# print(grid_search_cv.best_score_)
# print(grid_search_cv.best_params_)



model = RandomForestClassifier(verbose=2, n_jobs=-1, max_depth= None, 
                                            max_features= 5, min_samples_split= 2, n_estimators= 500)
model.fit(x_train,y_train)

y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)

# Training Set Performance
model_train_accuracy = accuracy_score(y_train, y_train_pred)
model_train_f1 = f1_score(y_train, y_train_pred, average="weighted")
model_train_precision = precision_score(y_train, y_train_pred, average="weighted")
model_train_recall = recall_score(y_train, y_train_pred, average="weighted")

# Test Set Performance
model_test_accuracy = accuracy_score(y_test, y_test_pred)
model_test_f1 = f1_score(y_test, y_test_pred, average="weighted")
model_test_precision = precision_score(y_test, y_test_pred, average="weighted")
model_test_recall = recall_score(y_test, y_test_pred, average="weighted")

print("Random Forest")
print("Model Performance for Training set")
print(" - Accuracy: {:.4f}".format(model_train_accuracy))
print(" - F1 score: {:.4f}".format(model_train_f1))
print(" - Precision: {:.4f}".format(model_train_precision))
print(" - Recall: {:.4f}".format(model_train_recall))
# print(" - Roc Auc Score: {:.4f}".format(model_train_rocauc_score))

print("--------------------------------------------------")
print("Model Performance for Test set")
print(" - Accuracy: {:.4f}".format(model_test_accuracy))
print(" - F1 score: {:.4f}".format(model_test_f1))
print(" - Precision: {:.4f}".format(model_test_precision))
print(" - Recall: {:.4f}".format(model_test_recall))
# print(" - Roc Auc Score: {:.4.f}".format(model_test_rocauc_score))


