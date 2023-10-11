from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb
import pandas as pd

class ModelBuild:
    def __init__(self, X_train, X_test, y_train, y_test, experimental_management:bool) -> None:
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.experimental_management = experimental_management

    def modelling(self):
        self.model.fit(self.X_train, self.y_train)
        y_train_pred = self.model.predict(self.X_train)

        self.model.fit(self.X_test, self.y_test)
        y_test_pred = self.model.predict(self.X_test)

        return y_train_pred, y_test_pred
    
    def process_hyperparameter(self, x,  y):
        if self.hyperparameter!=None:
            random_search = RandomizedSearchCV(estimator=self.model, param_distributions=self.hyperparameter,cv=5, 
                                            verbose=3,n_jobs=-1)
            from datetime import datetime
            def timer(start_time=None):
                if not start_time:
                    start_time = datetime.now()
                    return start_time
                elif start_time:
                    thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
                    tmin, tsec = divmod(temp_sec, 60)
                    print("\n Time taken: %i hours %i minutes and %s seconds." % (thour, tmin, round(tsec,2)))
            start_time = timer(None)
            random_search.fit(x, y) 
            start_time = timer(start_time)
            print(random_search.best_estimator_)
            print(random_search.best_params_)
            best_estimator = random_search.best_estimator_

class LogRegression(ModelBuild):
    def __init__(self, X_train, X_test, y_train, y_test,  experimental_management):
        self.model = LogisticRegression()
        self.hyperparameter = None
        super().__init__(X_train, X_test, y_train, y_test,  experimental_management)

class RandForest(ModelBuild):
    def __init__(self, X_train, X_test, y_train, y_test,  experimental_management):
        self.model = RandomForestClassifier()
        self.hyperparameter = None
        super().__init__(X_train, X_test, y_train, y_test,  experimental_management)

class XgBoost(ModelBuild):
    def __init__(self, X_train, X_test, y_train, y_test,  experimental_management):
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