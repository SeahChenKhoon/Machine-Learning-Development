import numpy as np
import pandas as pd
import Visualisation
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class Model_Build():
    def __init__(self, dataframe:pd.DataFrame, target_variable, hyperparameters, test_size, random_state, is_notebook) -> None:
        self.dataframe = dataframe
        self.target_variable = target_variable
        self.hyperparameters = hyperparameters
        self.test_size = test_size
        self.random_state = random_state
        self.is_notebook = is_notebook
        return None

    def prepare_data(self):
        X = self.dataframe.drop([self.target_variable], axis=1)
        y = self.dataframe[self.target_variable]
        return X, y
    
    def model_processing(self, model):
        X, y = self.prepare_data()
        X_train, X_test, y_train, y_test = self.train_test_split(X, y, self.test_size, self.random_state)
        X_train_smote, y_train_smote = self.SMOTE(X_train, y_train, self.random_state)
        X_train = X_train_smote
        y_train = y_train_smote
        # X_train, X_test = self.min_max_scaler(X_train, X_test)
        X_train, X_test = self.standard_scaler(X_train, X_test)
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        self.model_rpt_print(y_train, y_train_pred, y_test, y_test_pred, self.is_notebook)
        return None

    def SMOTE(self, X_train, y_train, random_state):
        smt = SMOTE(random_state=random_state)
        os_data_X,os_data_y=smt.fit_resample(X_train, y_train)
        return os_data_X, os_data_y

    def train_test_split(self, X, y, test_size, random_state):
        return train_test_split(X, y, test_size=test_size , random_state=random_state, stratify=y)
    
    def min_max_scaler(self, X_train, X_test):
        n_scaler = MinMaxScaler()
        X_train_scaled = n_scaler.fit_transform(X_train.astype(float))
        X_test_scaled = n_scaler.transform(X_test.astype(float))
        return X_train_scaled, X_test_scaled

    def standard_scaler(self, X_train, X_test):
        n_scaler = StandardScaler()
        X_train_scaled = n_scaler.fit_transform(X_train.astype(float))
        X_test_scaled = n_scaler.transform(X_test.astype(float))
        return X_train_scaled, X_test_scaled

    def prt_classification_rpt(self, prt_label, y_actual, y_pred):
        print("\033[1m" + prt_label +" \033[0m")
        print(classification_report(y_actual, y_pred))
        return None

    def prt_perf_score(self, prt_label, y_actual, y_pred):
        print("\033[1m" + prt_label +" \033[0m")
        print("Test Accuracy:",format(metrics.accuracy_score(y_actual, y_pred), '.4f'))
        print("Test Precision:",format(metrics.precision_score(y_actual, y_pred,average='micro'), '.4f'))
        print("Test Recall:",format(metrics.recall_score(y_actual, y_pred,average='micro'), '.4f')) 
        return None

    def model_rpt_print(self, y_train, y_train_pred, y_test, y_test_pred, is_notebook):
        print("\033[1mClassification Report \033[0m")
        self.prt_classification_rpt("Train", y_train, y_train_pred)
        self.prt_classification_rpt("Test", y_test, y_test_pred)
        print("")
        print("\033[1mConfusion Metric\033[0m")
        Visualisation.vs_confusion_matrix("Train", y_train, y_train_pred, is_notebook)
        Visualisation.vs_confusion_matrix("Test", y_test, y_test_pred, is_notebook)
        print("")
        print("\033[1mPerformance Metrics\033[0m")
        self.prt_perf_score("Train", y_train, y_train_pred)
        self.prt_perf_score("Test", y_test, y_test_pred)
        return None


class Logistic_Regression(Model_Build):
    # RFE and Logit
    def model_processing(self):
        lr = LogisticRegression(**self.hyperparameters)
        super().model_processing(lr)
        return None

# class Decision_Tree(Model_Build):
#     def model_processing(self, X, y, test_size, random_state, hyperparameters, is_notebook):
#         dtc = DecisionTreeClassifier(**hyperparameters)

#         X_train, X_test, y_train, y_test = self.train_test_split(X, y, test_size, random_state)
#         X_train_smote, y_train_smote = self.SMOTE(X_train, y_train, random_state)
#         X_train = X_train_smote
#         y_train = y_train_smote
#         dtc.fit(X_train, y_train)
#         dot_data = export_graphviz(X_train, out_file=None, feature_names=list(X_train.columns.values), 
#                                 class_names=[0, 1, 2], rounded=True, filled=True)
#         y_train_pred = dtc.predict(X_train)
#         y_test_pred = dtc.predict(X_test)
#         self.model_rpt_print(y_train, y_train_pred, y_test, y_test_pred, is_notebook)
#         return None


# hyperparameter_dict = {
#     'solver': 'liblinear'  # Specify the solver algorithm
# }
# train_model("Initial", X, y, hyperparameter_dict,True, False)
# def train_model(label, X, y, hyperparameter, output_label, return_result):
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
#     logreg = LogisticRegression(**hyperparameter) 
#     logreg.fit(X_train, y_train)
    
#     y_predict_train = logreg.predict(X_train)
#     y_pred = logreg.predict(X_test)

#     confusion_matrix_train = confusion_matrix(y_train, y_predict_train)
#     confusion_matrix_test = confusion_matrix(y_test, y_pred)
#     class_rpt_test = classification_report(y_test, y_pred)
#     class_rpt_train = classification_report(y_train, y_predict_train)
    
#     if output_label == True:
#         print("\033[1m" + label +" (Heart) \033[0m")
#         print('Confusion_matrix - Train')
#         print(confusion_matrix_train)

#         print('Accuracy of logistic regression classifier on Training set: {:.4f}\n'.format(logreg.score(X_train, y_train)))

#         print('\nClassification Report - Train')
#         print(class_rpt_train)
#         print('\n\n\n')
        
#         print('Confusion_matrix - Test')
#         print(confusion_matrix_test)

#         print('Accuracy of logistic regression classifier on test set: {:.4f}\n'.format(logreg.score(X_test, y_test)))

#         print('\nClassification Report - Test')
#         print(class_rpt_test)
#         print('\n\n\n')
        
        
#     if return_result == True:
#         return logreg.score(X_test, y_test), confusion_matrix_1, classification_report(y_test, y_pred)
#     else:
#         return None

# df_final_vars=df_final.columns.values.tolist()
# y=['y']
# X=[i for i in df_final_vars if i not in y]
# from sklearn.feature_selection import RFE
# from sklearn.linear_model import LogisticRegression
# logreg = LogisticRegression()
# rfe = RFE(estimator=LogisticRegression(), n_features_to_select=20)
# rfe = rfe.fit(os_data_X, os_data_y.values.ravel())
# print(rfe.support_)
# print(rfe.ranking_)

# X = df_final.loc[:, df_final.columns != 'y']
# y = df_final.loc[:, df_final.columns == 'y']
# from imblearn.over_sampling import SMOTE
# os = SMOTE(random_state=0)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
# columns = X_train.columns
# os_data_X,os_data_y=os.fit_resample(X_train, y_train)
# os_data_X = pd.DataFrame(data=os_data_X,columns=columns )
# os_data_y= pd.DataFrame(data=os_data_y,columns=['y'])
# # we can Check the numbers of our data
# print("length of oversampled data is ",len(os_data_X))
# print("Number of no subscription in oversampled data",len(os_data_y[os_data_y['y']==0]))
# print("Number of subscription",len(os_data_y[os_data_y['y']==1]))
# print("Proportion of no subscription data in oversampled data is ",len(os_data_y[os_data_y['y']==0])/len(os_data_X))
# print("Proportion of subscription data in oversampled data is ",len(os_data_y[os_data_y['y']==1])/len(os_data_X))














# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import RandomizedSearchCV
# import xgboost as xgb
# import pandas as pd
# import numpy as np
# import util

# class ModelBuild:
#     def __init__(self, model_name, X_train:np.ndarray, X_test:np.ndarray, y_train:pd.Series, y_test:pd.Series, 
#                   hyperparameter_enabled:bool, hyperparameter_tuning:list) -> None:
#         """
#         """
#         self.model_name =  model_name
#         self.X_train:np.ndarray = X_train
#         self.X_test:np.ndarray = X_test
#         self.y_train:pd.Series = y_train
#         self.y_test:pd.Series = y_test
#         self.hyperparameter_enabled:bool = hyperparameter_enabled
#         if hyperparameter_tuning == None:
#             self.hyperparameter_tuning = None
#         else:
#             self.hyperparameter_tuning:dict = dict()
#             for key, values in hyperparameter_tuning.items():
#                 self.hyperparameter_tuning[key] = values
#         return None

#     def modelling(self)->tuple[np.ndarray, np.ndarray]:
#         """
#         Train the model and make predictions on training and testing data.

#         This method fits the model to the training data, makes predictions on both the training
#         and testing data, and returns the predicted values for training and testing sets.

#         Returns:
#             Tuple[np.ndarray, np.ndarray]: A tuple containing two NumPy arrays:
#                 - y_train_pred: Predicted values for the training set.
#                 - y_test_pred: Predicted values for the testing set.
#         """
#         self.model.fit(self.X_train, self.y_train)
#         y_train_pred = self.model.predict(self.X_train)

#         self.model.fit(self.X_test, self.y_test)
#         y_test_pred = self.model.predict(self.X_test)

#         return y_train_pred, y_test_pred
    
#     def process_hyperparameter(self, x:np.ndarray,  y:pd.Series)->None:
#         """
#         Perform hyperparameter tuning using Randomized Search Cross-Validation.

#         This method conducts hyperparameter tuning using Randomized Search Cross-Validation,
#         where it searches for the best hyperparameters for the model. It prints the best estimator
#         and best parameters found during the search.

#         Parameters:
#             x (np.ndarray): The feature matrix.
#             y (pd.Series): The target variable.

#         Returns:
#             None: This method does not return a value.
#         """

#         if self.hyperparameter_enabled==True and self.hyperparameter_tuning!=None:
#             random_search = RandomizedSearchCV(estimator=self.model, param_distributions=self.hyperparameter_tuning,cv=5, 
#                                             verbose=3,n_jobs=-1)
#             start_time = util.timer(None)
#             random_search.fit(x, y) 
#             start_time = util.timer(start_time)
#             print(random_search.best_estimator_)
#             print(random_search.best_params_)
#         return None

# class LogRegression(ModelBuild):
#     def __init__(self, model_name, X_train:np.ndarray, X_test:np.ndarray, y_train:pd.Series, 
#                  y_test:pd.Series, hyperparameter_enabled:bool, hyperparameter_tuning:list)->None:
#         """
#         Initialize an instance of the class.
#         """
#         self.model = LogisticRegression()
#         self.hyperparameter = None
#         super().__init__(model_name, X_train, X_test, y_train, y_test,  hyperparameter_enabled, hyperparameter_tuning)
#         return None

# class RandForest(ModelBuild):
#     def __init__(self, model_name, X_train:np.ndarray, X_test:np.ndarray, y_train:pd.Series, 
#                  y_test:pd.Series,  hyperparameter_enabled:bool, hyperparameter_tuning:list):
#         """
#         Initialize an instance of the class.
#         """
#         self.model = RandomForestClassifier()
#         self.hyperparameter = None
#         super().__init__(model_name, X_train, X_test, y_train, y_test,  hyperparameter_enabled, hyperparameter_tuning)
#         return None

# class XgBoost(ModelBuild):
#     def __init__(self, model_name, X_train:np.ndarray, X_test:np.ndarray, y_train:pd.Series, 
#                  y_test:pd.Series, hyperparameter_enabled:bool, hyperparameter_tuning:list):
#         """
#         Initialize an instance of the class.
#         """
#         self.model = xgb.XGBClassifier()
#         super().__init__(model_name, X_train, X_test, y_train, y_test,  hyperparameter_enabled, hyperparameter_tuning)
#         return None