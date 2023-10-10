import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay, \
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve

class ModelEval:
    def __init__(self, model_type:str, y_train:pd.DataFrame, y_train_pred:pd.DataFrame) -> None:
        self.model_type = model_type.upper()
        self.model_train_accuracy = accuracy_score(y_train, y_train_pred)
        self.model_train_f1 = f1_score(y_train, y_train_pred, average="weighted")
        self.model_train_precision = precision_score(y_train, y_train_pred, average="weighted")
        self.model_train_recall = recall_score(y_train, y_train_pred, average="weighted")

    def print_report(self, report_type="Training"):
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
