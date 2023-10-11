from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb
import pandas as pd
import numpy as np
import util

class ModelBuild:
    def __init__(self, X_train:np.ndarray, X_test:np.ndarray, y_train:pd.Series, y_test:pd.Series, 
                 experimental_management:bool) -> None:
        """
        Initialize an instance of the class.

        This constructor initializes an instance of the class with training and testing data,
        as well as an experimental management flag.

        Parameters:
            X_train (np.ndarray): The feature matrix for the training set.
            X_test (np.ndarray): The feature matrix for the testing set.
            y_train (pd.Series): The target variable for the training set.
            y_test (pd.Series): The target variable for the testing set.
            experimental_management (bool): A flag indicating whether experimental management is enabled.

        Returns:
            None: This constructor does not return a value.
        """
        self.X_train:np.ndarray = X_train
        self.X_test:np.ndarray = X_test
        self.y_train:pd.Series = y_train
        self.y_test:pd.Series = y_test
        self.experimental_management:bool = experimental_management
        return None

    def modelling(self)->tuple[np.ndarray, np.ndarray]:
        """
        Train the model and make predictions on training and testing data.

        This method fits the model to the training data, makes predictions on both the training
        and testing data, and returns the predicted values for training and testing sets.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing two NumPy arrays:
                - y_train_pred: Predicted values for the training set.
                - y_test_pred: Predicted values for the testing set.
        """
        self.model.fit(self.X_train, self.y_train)
        y_train_pred = self.model.predict(self.X_train)

        self.model.fit(self.X_test, self.y_test)
        y_test_pred = self.model.predict(self.X_test)

        return y_train_pred, y_test_pred
    
    def process_hyperparameter(self, x:np.ndarray,  y:pd.Series)->None:
        """
        Perform hyperparameter tuning using Randomized Search Cross-Validation.

        This method conducts hyperparameter tuning using Randomized Search Cross-Validation,
        where it searches for the best hyperparameters for the model. It prints the best estimator
        and best parameters found during the search.

        Parameters:
            x (np.ndarray): The feature matrix.
            y (pd.Series): The target variable.

        Returns:
            None: This method does not return a value.
        """
        if self.hyperparameter!=None:
            random_search = RandomizedSearchCV(estimator=self.model, param_distributions=self.hyperparameter,cv=5, 
                                            verbose=3,n_jobs=-1)
            start_time = util.timer(None)
            random_search.fit(x, y) 
            start_time = util.timer(start_time)
            print(random_search.best_estimator_)
            print(random_search.best_params_)
        return None

class LogRegression(ModelBuild):
    def __init__(self, X_train:np.ndarray, X_test:np.ndarray, y_train:pd.Series, 
                 y_test:pd.Series, experimental_management:bool)->None:
        """
        Initialize an instance of the class.
        """
        self.model = LogisticRegression()
        self.hyperparameter = None
        super().__init__(X_train, X_test, y_train, y_test,  experimental_management)
        return None

class RandForest(ModelBuild):
    def __init__(self, X_train, X_test, y_train, y_test,  experimental_management):
        """
        Initialize an instance of the class.
        """
        self.model = RandomForestClassifier()
        self.hyperparameter = None
        super().__init__(X_train, X_test, y_train, y_test,  experimental_management)
        return None

class XgBoost(ModelBuild):
    def __init__(self, X_train, X_test, y_train, y_test,  experimental_management):
        """
        Initialize an instance of the class.
        """
        self.model = xgb.XGBClassifier()
        self.hyperparameter = {"max_depth": [None,3,4,5,6,8, 10,12,15],
            "booster": ["gbtree"],
            "tree_method": ["auto","exact","approx","hist"],
            "alpha":[0,1,2,3,4,5],
            "lambda":[0,1,2,3,4,5],
            "colsample_bytree" : [0, 0.5, 1],
            "colsample_bylevel" : [0, 0.5, 1],
            "colsample_bynode" : [0, 0.5, 1],
            "grow_policy": ["depthwise","lossguide"],
            "learning_rate": [0.05, 0.10,0.15, 0.20, 0.25,0.30],
            "min_child_weight": [1,3,5,7],
            "gamma": [0.0,0.1,0.2,0.3,0.4]
            }
        super().__init__(X_train, X_test, y_train, y_test,  experimental_management)
        return None