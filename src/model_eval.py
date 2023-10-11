import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay, \
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve

class ModelEval:
    def __init__(self, model_type:str, y_train:pd.DataFrame, y_train_pred:pd.DataFrame) -> None:
        """
        Initialize an instance of the class.

        This constructor initializes an instance of the class with model type and performance metrics
        based on the training data.

        Parameters:
            model_type (str): The type of the model.
            y_train (pd.DataFrame): The true target values for the training data.
            y_train_pred (pd.DataFrame): The predicted target values for the training data.

        Returns:
            None: This constructor does not return a value.
        """
        self.model_type = model_type.upper()
        self.model_train_accuracy = accuracy_score(y_train, y_train_pred)
        self.model_train_f1 = f1_score(y_train, y_train_pred, average="weighted")
        self.model_train_precision = precision_score(y_train, y_train_pred, average="weighted")
        self.model_train_recall = recall_score(y_train, y_train_pred, average="weighted")
        return None

    def print_report(self, report_type:str="Training")-> None:
        """
        Print a performance report for the model.

        This method prints a performance report for the model, including accuracy, F1 score,
        precision, and recall.

        Parameters:
            report_type (str, optional): The type of report to print, "Training" (default) or "Testing".

        Returns:
            None: This method does not return a value.
        """
        print(f"{self.model_type}")
        if report_type == "Training":
            print(f"Model Performance for Training set")
        else:
            print(f"Model Performance for Testing set")
        print(" - Accuracy: {:.4f}".format(self.model_train_accuracy))
        print(" - F1 score: {:.4f}".format(self.model_train_f1))
        print(" - Precision: {:.4f}".format(self.model_train_precision))
        print(" - Recall: {:.4f}".format(self.model_train_recall))
        print('='*35)
        print('\n')
        return None
