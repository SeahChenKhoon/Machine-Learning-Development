import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score

class Modelling:
    
    def __init__(self, dataframe:pd.DataFrame, target_col:str, test_size:float)->None:
        X = dataframe.drop([target_col], axis=1)
        y = dataframe[target_col]
        self._x = X
        self._y = y
        if test_size != None:
            self._test_size = test_size
        else:
            self._test_size = None

    def model_predict(self, X_test:pd.DataFrame, model, predict_col:str):
        return model.predict(X_test)

    def train_impute_model(self, model,X_train, y_train, X_impute):
        model.fit(X_train, y_train)
        return model.predict(X_impute)

    def train_model(self, model):
        x_train, x_test, y_train, y_test = train_test_split(self._x, self._y, test_size=self._test_size, random_state=42)
        model.fit(x_train, y_train)
        accuracy = model.score(x_test, y_test)

        score = cross_val_score(model, self._x, self._y, cv=5)
        cv_score = np.mean(score)
        return {accuracy,cv_score, model}
    
class RandomForestClassify(Modelling):
    def model(self):
        return RandomForestClassifier()

class LogisticRegression(Modelling):
    def model(self):
        return LogisticRegression()
