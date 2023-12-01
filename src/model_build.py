from sklearn.model_selection import train_test_split
def train_test_split(X, y, test_size, random_state):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size , random_state=random_state)

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