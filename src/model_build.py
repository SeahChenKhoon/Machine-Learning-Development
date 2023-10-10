from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import pandas as pd

class ModelBuild:
    def __init__(self, model_type='logistic_regression'):
        if model_type == 'logistic_regression':
            self.model = LogisticRegression()
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier()
        elif model_type == 'XG Boost':
            self.model = xgb.XGBClassifier()  
        else:
            raise ValueError("Invalid model_type. Supported values: \
                             'logistic_regression', 'random_forest'")
    
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
    
    def predict(self, X_test):
        return self.model.predict(X_test)


