import numpy as np
import pandas as pd
import Visualisation
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

class Model_Build():
    def __init__(self, dataframe:pd.DataFrame, target_variable, hyperparameters, test_size, random_state, is_notebook) -> None:
        self.dataframe = dataframe
        self.target_variable = target_variable
        self.hyperparameters = hyperparameters
        self.test_size = test_size
        self.random_state = random_state
        self.is_notebook = is_notebook
        self.X = None
        self.y = None
        return None


    def prepare_data(self):
        X = self.dataframe.drop([self.target_variable], axis=1)
        y = self.dataframe[self.target_variable]
        self.X = X
        self.y = y
        return self.train_test_split(self.X, self.y, self.test_size, self.random_state)
    
    def model_processing(self, model):
        X_train, X_test, y_train, y_test =  self.prepare_data()
        X_train_smote, y_train_smote = self.SMOTE(X_train, y_train, self.random_state)
        X_train = X_train_smote
        y_train = y_train_smote
        X_test_smote, y_test_smote = self.SMOTE(X_test, y_test, self.random_state)
        X_test = X_test_smote
        y_test = y_test_smote
        # X_train, X_test = self.min_max_scaler(X_train, X_test)
        X_train, X_test = self.standard_scaler(X_train, X_test)
        model_train = model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        self.model_rpt_print(y_train, y_train_pred, y_test, y_test_pred, self.is_notebook)
        return model_train


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

    def return_X(self):
        return self.X
    
    def GridSearchCV(self, model, param_grid, verbose):
        grid = GridSearchCV(model,  param_grid, verbose = verbose)
        X_train, X_test, y_train, y_test =  self.prepare_data()
        grid.fit(X_train,y_train)
        print(grid.best_estimator_)
        return grid
    
    def RandomizedSearchCV(self, model: any,
                     param_grid: dict,verbose,
        ) -> None:
        # Define RandomSearchCV
        random_search = RandomizedSearchCV(
            model,
            param_distributions=param_grid,
            n_iter=50,  # Specify the number of iterations here
            scoring='balanced_accuracy',
            cv=3,
            random_state=self.random_state,
            verbose=verbose
        )
        X_train, X_test, y_train, y_test =  self.prepare_data()

        # Fit random_search
        random_search.fit(X_train, y_train)
        print("Random Search Best Params")
        print(random_search.best_params_)
        print("Random Search Best Estimators")
        print(random_search.best_estimator_)
        # Get best params and model
        return (random_search.best_params_)

class Logistic_Regression(Model_Build):
    # RFE and Logit
    def model_processing(self):
        lr = LogisticRegression(**self.hyperparameters)
        super().model_processing(lr)
        return None
    
    def GridSearchCV(self, param_grid, verbose):
        lr = LogisticRegression(**self.hyperparameters)
        grid = super().GridSearchCV(lr, param_grid, verbose)
        super().model_processing(grid)

    def RandomizedSearchCV(self, param_grid, verbose):
        lr = SVC(**self.hyperparameters)
        grid = super().RandomizedSearchCV(lr, param_grid, verbose)     
        super().model_processing(grid)
        return None

class Decision_Tree_Classifier(Model_Build):
    def model_processing(self):
        dtc = DecisionTreeClassifier(**self.hyperparameters)
        dtc_train = super().model_processing(dtc)
        return dtc_train

    def GridSearchCV(self, param_grid, verbose):
        dtc = DecisionTreeClassifier(**self.hyperparameters)
        grid = super().GridSearchCV(dtc, param_grid, verbose)     
        super().model_processing(grid)
        return None
    
    def RandomizedSearchCV(self, param_grid, verbose):
        dtc = DecisionTreeClassifier(**self.hyperparameters)
        grid = super().RandomizedSearchCV(dtc, param_grid, verbose)     
        # super().model_processing(grid)
        return None
    
class Random_Forest_Classifier(Model_Build):
    def model_processing(self):
        rfc = RandomForestClassifier(**self.hyperparameters)
        super().model_processing(rfc)
        return None

    def GridSearchCV(self, param_grid, verbose):
        rfc = RandomForestClassifier(**self.hyperparameters)
        grid = super().GridSearchCV(rfc, param_grid, verbose)     
        super().model_processing(grid)
        return None

    def RandomizedSearchCV(self, param_grid, verbose):
        rfc = RandomForestClassifier(**self.hyperparameters)
        grid = super().RandomizedSearchCV(rfc, param_grid, verbose)     
        return None

class Support_Vector_Classifier(Model_Build):
    def model_processing(self):
        svc = SVC(**self.hyperparameters)
        super().model_processing(svc)
        return None

    def GridSearchCV(self, param_grid, verbose):
        svc = SVC(**self.hyperparameters)
        grid = super().GridSearchCV(svc, param_grid, verbose)     
        super().model_processing(grid)
        return None
    
    def RandomizedSearchCV(self, param_grid, verbose):
        svc = SVC(**self.hyperparameters)
        grid = super().RandomizedSearchCV(svc, param_grid, verbose)     
        super().model_processing(grid)
        return None

class Gradient_Boosting_Classifier(Model_Build):
    def model_processing(self):
        gbc = GradientBoostingClassifier(**self.hyperparameters)
        super().model_processing(gbc)
        return None
    
    def GridSearchCV(self, param_grid, verbose):
        gbc = GradientBoostingClassifier(**self.hyperparameters)
        grid = super().GridSearchCV(gbc, param_grid, verbose)     
        super().model_processing(grid)
        return None
    
    def RandomizedSearchCV(self, param_grid, verbose):
        gbc = GradientBoostingClassifier(**self.hyperparameters)
        super().RandomizedSearchCV(gbc, param_grid, verbose)     
        return None