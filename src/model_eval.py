import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay, \
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix

class ModelEval:
    def __init__(self, model_type:str, y:pd.DataFrame, y_pred:pd.DataFrame) -> None:
        """
        Initializes an instance of the class with the specified parameters.

        Parameters:
        - model_type (str): The type of the model as a string.
        - y (pd.DataFrame): The actual labels (ground truth) as a pandas DataFrame.
        - y_pred (pd.DataFrame): The predicted labels as a pandas DataFrame.

        Returns:
        None
        """
        self.model_type = model_type.upper()

        self.confusion_matrix_test = confusion_matrix(y, y_pred)
        self.class_rpt_test = classification_report(y, y_pred)
        return None

    def print_report(self, report_type:str="Training")-> None:
        """
        Print the performance report for the specified set (Training or Testing).

        Parameters:
        - report_type (str, optional): The type of the report to be printed. Defaults to "Training".

        Returns:
        None
        """
        print(f"{self.model_type}")
        if report_type == "Training":
            print(f"Model Performance for Training set")
        else:
            print(f"Model Performance for Testing set")
        print('Confusion_matrix')
        print(self.confusion_matrix_test)
        print('\nClassification Report')
        print(self.class_rpt_test)
        print('='*35)
        print('\n')
        return None
