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

class ModelBuild():
    def prepare_data(self, df, target_variable, test_size, random_state):
        X = df.drop([target_variable], axis=1)
        y = df[target_variable]
        return self.train_test_split(X, y, test_size, random_state)
    
    def SMOTE(self, X_train, y_train, random_state):
        smt = SMOTE(random_state=random_state)
        os_data_X,os_data_y=smt.fit_resample(X_train, y_train)
        return os_data_X, os_data_y
    
    def model_processing(self, model, X_train, y_train, X_test):
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        return y_train_pred, y_test_pred

    def train_test_split(self, X, y, test_size, random_state):
        return train_test_split(X, y, test_size=test_size , random_state=random_state, stratify=y)

    def normalised_data(self, X_train, X_test, scalar_option, col_to_scale):
        X_train_normalised = X_train.copy()
        X_test_normalised = X_test.copy()
        if scalar_option=="MinMaxScaler":
            n_scaler = MinMaxScaler()
        else:
            n_scaler = StandardScaler()
        X_train_scaled = n_scaler.fit_transform(X_train[col_to_scale])
        X_test_scaled = n_scaler.transform(X_test[col_to_scale])
        X_train_normalised[col_to_scale] = X_train_scaled
        X_test_normalised[col_to_scale] = X_test_scaled
        return X_train_normalised, X_test_normalised

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

    def model_rpt_print(self, y_train, y_train_pred, y_test, y_test_pred):
        print("\033[1mClassification Report \033[0m")
        self.prt_classification_rpt("Train", y_train, y_train_pred)
        self.prt_classification_rpt("Test", y_test, y_test_pred)
        print("")
        print("\033[1mConfusion Metric\033[0m")
        Visualisation.vs_confusion_matrix("Train", y_train, y_train_pred)
        Visualisation.vs_confusion_matrix("Test", y_test, y_test_pred)
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

class Logistic_Regression(ModelBuild):
    # RFE and Logit
    def model_processing(self, X_train, y_train, X_test, hyperparameters):
        lr = LogisticRegression(**hyperparameters)
        return super().model_processing(lr,X_train, y_train, X_test)
    
    # def GridSearchCV(self, param_grid, verbose):
    #     lr = LogisticRegression(**self.hyperparameters)
    #     grid = super().GridSearchCV(lr, param_grid, verbose)
    #     super().model_processing(grid)

    # def RandomizedSearchCV(self, param_grid, verbose):
    #     lr = SVC(**self.hyperparameters)
    #     grid = super().RandomizedSearchCV(lr, param_grid, verbose)     
    #     super().model_processing(grid)
    #     return None

class Decision_Tree_Classifier(ModelBuild):
    def model_processing(self, X_train, y_train, X_test, hyperparameters):
        dtc = DecisionTreeClassifier(**hyperparameters)
        return super().model_processing(dtc,X_train, y_train, X_test)

    # def GridSearchCV(self, param_grid, verbose):
    #     dtc = DecisionTreeClassifier(**self.hyperparameters)
    #     grid = super().GridSearchCV(dtc, param_grid, verbose)     
    #     super().model_processing(grid)
    #     return None
    
    # def RandomizedSearchCV(self, param_grid, verbose):
    #     dtc = DecisionTreeClassifier(**self.hyperparameters)
    #     grid = super().RandomizedSearchCV(dtc, param_grid, verbose)     
    #     # super().model_processing(grid)
    #     return None
    
class Random_Forest_Classifier(ModelBuild):
    def model_processing(self, X_train, y_train, X_test,hyperparameters):
        rfc = RandomForestClassifier(**hyperparameters)
        return super().model_processing(rfc, X_train, y_train, X_test)

    def GridSearchCV(self, param_grid, verbose):
        rfc = RandomForestClassifier(**self.hyperparameters)
        grid = super().GridSearchCV(rfc, param_grid, verbose)     
        super().model_processing(grid)
        return None

    def RandomizedSearchCV(self, param_grid, verbose):
        rfc = RandomForestClassifier(**self.hyperparameters)
        return super().RandomizedSearchCV(rfc, param_grid, verbose)     

class Support_Vector_Classifier(ModelBuild):
    def model_processing(self, X_train, y_train, X_test,hyperparameters):
        svc = SVC(**hyperparameters)
        return super().model_processing(svc, X_train, y_train, X_test)

    # def GridSearchCV(self, param_grid, verbose):
    #     svc = SVC(**self.hyperparameters)
    #     grid = super().GridSearchCV(svc, param_grid, verbose)     
    #     super().model_processing(grid)
    #     return None
    
    # def RandomizedSearchCV(self, param_grid, verbose):
    #     svc = SVC(**self.hyperparameters)
    #     grid = super().RandomizedSearchCV(svc, param_grid, verbose)     
    #     super().model_processing(grid)
    #     return None

class Gradient_Boosting_Classifier(ModelBuild):
    def model_processing(self, X_train, y_train, X_test,hyperparameters):
        gbc = GradientBoostingClassifier(**hyperparameters)
        return super().model_processing(gbc, X_train, y_train, X_test)
    
    # def GridSearchCV(self, param_grid, verbose):
    #     gbc = GradientBoostingClassifier(**self.hyperparameters)
    #     grid = super().GridSearchCV(gbc, param_grid, verbose)     
    #     super().model_processing(grid)
    #     return None
    
    # def RandomizedSearchCV(self, param_grid, verbose):
    #     gbc = GradientBoostingClassifier(**self.hyperparameters)
    #     super().RandomizedSearchCV(gbc, param_grid, verbose)     
    #     return None