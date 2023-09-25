import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score

class Modelling:
    
    def __init__(self, dataframe:pd.DataFrame, target_col:str, test_size=0.2)->None:
        X = dataframe.drop([target_col], axis=1)
        y = dataframe[target_col]
        self._x = X
        self._y = y
        self._test_size = test_size

    def model_result(self, model):
        x_train, x_test, y_train, y_test = train_test_split(self._x, self._y, test_size=self._test_size, random_state=42)
        model.fit(x_train, y_train)
        accuracy = model.score(x_test, y_test)

        score = cross_val_score(model, self._x, self._y, cv=5)
        cv_score = np.mean(score)
        return {accuracy,cv_score}
    
class RandomForestClassify(Modelling):
    def model(self):
        return RandomForestClassifier()


