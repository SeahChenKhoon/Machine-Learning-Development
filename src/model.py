import pandas as pd
import datetime
import util
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report,accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix

def print_std_model_evaulation_rpt(model, Y_test, Y_pred, initialise=False):
    report = classification_report(Y_test, Y_pred, output_dict=True)
    weighted_accuracy = accuracy_score(Y_test, Y_pred)
    if initialise:
        metrics = {'Model': [model],
            'Recall': [report['weighted avg']['recall']],
            'Precision' : [report['weighted avg']["precision"]],
            'Accuracy' : [weighted_accuracy],
            'f1_score' : [report['weighted avg']["f1-score"]],
            'MAE' : [mean_absolute_error(Y_test,Y_pred)],
            'MSE' : [mean_squared_error(Y_test,Y_pred, squared=False)]
            }
    else:
        metrics = [model, report['weighted avg']['recall'],report['weighted avg']["precision"],
                   weighted_accuracy,report['weighted avg']["f1-score"], mean_absolute_error(Y_test,Y_pred), 
                   mean_squared_error(Y_test,Y_pred, squared=False)]
    return metrics

def training_model (dataframe:pd.DataFrame)->None:

    feature_list = ["Gender", "Onboard Wifi Service", "Embarkation/Disembarkation time convenient", "Ease of Online booking", 
                    "Gate location", "Onboard Dining Service", "Online Check-in", "Cabin Comfort", "Onboard Entertainment", 
                    "Cabin service", "Baggage handling", "Port Check-in Service", "Onboard Service", "Cleanliness", 
                    "Cruise Name", "Dining", "Distance in KM", "Age"]
    target_var = "Ticket Type"

    X = dataframe[feature_list]
    y = dataframe[target_var]
    dt = datetime.datetime.now()
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, random_state = 1)
    model_params = {
        "decision_tree": {
            "model": DecisionTreeClassifier(),
            "params": {
                "max_depth": [1280],
                "max_leaf_nodes": [1280],
                "splitter": ["best"],
                "max_features": [50],
                "min_impurity_decrease": [0.5],
                "class_weight":["balanced"]
            }
            # Summary:
            # "max_depth" : The bigger the number the better the mean_test_score -> df_result_decision_tree_01_max_depth_max_max_leaf_nodes.xlsx
            # "max_leaf_nodes": The bigger the number the better the mean_test_score -> df_result_decision_tree_01_max_depth_max_max_leaf_nodes.xlsx
            # "splitter": "best" perform better than random so "best" selected ->df_result_decision_tree_02_splitter.xlsx
            # "max_features": int seems to fair better than others -> df_result_decision_tree_03_splitter.xlsx
            # "max_features": mean_test_score seems to reach peak at 50 and drops. put more values to support. -> df_result_decision_tree_04_max_features.xlsx
            # "max_features": From above observation, i fix at 50. -> df_result_decision_tree_06_class_weight
            # "min_impurity_decrease": 0.2 to 0.9 has same value of 0.478781905, I set 0.5 -> df_result_decision_tree_06_class_weight
            # "class_weight": No output for auto, so I set to balanced. -> df_result_decision_tree_06_class_weight
        },        
        "knn": {
            "model": KNeighborsClassifier(),
            "params": {
                "n_neighbors": [15],
                "weights": ['distance'], 
                "metric": ['euclidean'],
                "algorithm" : ['ball_tree'],
                "leaf_size" : [200],
                "p": [1],
                "metric": ["cityblock"]
            }
            # Summary:
            # "n_neighbors": The higher the better -> df_result_knn_01_Metric_Compare.xlsx
            # "metric" : There is no difference between using minkowski or euclidean, so I fixed at euclidean -> df_result_knn_01_Metric_Compare.xlsx
            # "weights" : Distance fair better than uniform -> df_result_knn_01_Metric_Compare.xls
            # "algorithm": 'ball_tree' seems to perform better than the rest -> df_result_knn_02_algorithm.xlsx
            # "p": p has a range of 1 to 3 but, does not seems to impact mean_test_score, so I set to 1 -> df_result_knn_04_p.xlsx
            # "leaf_size": 20 and 30 same, 40, 50 same value so increase more and see what happens -> df_result_knn_03_leaf_size.xlsx
            # "leaf_size": mean_test_score peaks at 200 and it drops so I fixed this value at 200 -> df_result_knn_04_p.xlsx 
            # "metric": "cityblock", "l1" and "manhattan" have the same score of 0.737437175. I fix at cityblock -> df_result_knn_05_metric.xlsx
        },
        "random_forest": {
            "model": RandomForestClassifier(),
            "params": {
                'bootstrap': [False],
                'max_depth': [40],
                'n_estimators': [40],
                'criterion': ["gini"],
                'min_samples_split': [5],
                'max_features': ["log2"],
                'class_weight': ["balanced_subsample"],
                'warm_start': [False]
            }
            # Summary:
            # "bootstrap": False fair better than True -> df_result_random_forest_01_bootstrap.xlsx
            # "max_depth" : The higher the better -> df_result_random_forest_01_bootstrap.xlsx
            # "n_estimators" : The higher the better -> df_result_random_forest_01_bootstrap.xlsx
            # "criterion": gini fair better than entropy, log_loss -> df_result_random_forest_02_criterion.xlsx
            # "min_samples_split": mean_test_score improves as values increases -> df_result_random_forest_03_min_samples_split.xlsx
            # "min_samples_split": Peak at 5 so i set to 5 -> df_result_random_forest_04_min_samples_split.xlsx
            # "max_features": log2 has the highest score -> df_result_random_forest_06_max_features.xlsx
            # "class_weight": balanced_subsample has higher than balanced -> df_result_random_forest_07_class_weight.xlsx
            # "warm_start": class_weight presets "balanced" or "balanced_subsample" are not recommended for warm_start if the fitted data differs from the full dataset.
        }
    }
    score=[]
    keys_list=[]
    for model_name, mp in model_params.items():
        dt = datetime.datetime.now()    
        keys_list = list(mp["params"])
        keys_list = ["param_" + element for element in keys_list]
        keys_list.append("mean_test_score")
        clf = GridSearchCV(mp["model"], param_grid=mp["params"], cv=5)
        clf.fit(X_train, Y_train)
        df_result = pd.DataFrame()
        df_result = pd.DataFrame(clf.cv_results_)
        util.output_csv(df_result[keys_list], "df_result_" + model_name)

        score.append({
            "model": model_name,
            "best_score": clf.best_score_,
            "best_params": clf.best_params_
        })
    df_score = pd.DataFrame(score, columns=["model","best_score","best_params"])
    util.output_csv(df_score,"df_best_score") 

