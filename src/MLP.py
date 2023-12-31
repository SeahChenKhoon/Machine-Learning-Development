# Importation of libraries
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from data_preprocessing import DataProcessing

from feature_engineering import FeatureEngineering
from model_build import ModelBuild
from model_build import Gradient_Boosting_Classifier

import ast
import pandas as pd
import util
import database


#Define the data path from src folder
DATA_PATH =  "../data/"
#Define the yaml path
YAML_FILEPATHNAME = "../config.yaml"
#Define position of pre_cruise database configuration in yaml file
PRE_CRUISE_DB = 0
POST_CRUISE_DB = 1
IS_NOTEBOOK = True

# Initialisation of Classes
dataprocessor = DataProcessing()
featureengineering = FeatureEngineering()
modelbuild = ModelBuild()

class ConvertNanToNone(BaseEstimator, TransformerMixin):
    """
    A custom scikit-learn transformer to replace NaN values with None in a dataset.

    This transformer fits to the data during training (fit method) and transforms
    the data by replacing NaN with None (transform method).

    Parameters:
        None

    Methods:
        fit(X, y=None): Fits the transformer to the data.
        transform(X): Transforms the data by replacing NaN with None.
    """
    def fit(self, X, y=None):
        """
        Fit method for the transformer. Since the transformer doesn't learn from the data,
        it simply returns itself.

        Parameters:
            X: The input data.
            y: The target labels (default=None).

        Returns:
            self
        """
        return self

    def transform(self, X):
        X = dataprocessor.replace_nan_none(X)
        return X
    
class SplitCompositeFields(BaseEstimator, TransformerMixin):
    def __init__(self, composite_fields_info) -> None:
        super().__init__()
        self.composite_fields_info = composite_fields_info

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = dataprocessor.split_composite_field(X, self.composite_fields_info)
        return X  

class RemoveColumnWithHighMissingVal(BaseEstimator, TransformerMixin):
    def __init__(self, missing_val_threshold) -> None:
        super().__init__()
        self.missing_val_threshold = missing_val_threshold
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = dataprocessor.rm_cols_high_missing(X, self.missing_val_threshold)
        return X  

class RemoveIDsCols(BaseEstimator, TransformerMixin):
    def __init__(self, id_cols) -> None:
        super().__init__()
        self.id_cols = id_cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = dataprocessor.rm_id_cols(X, self.id_cols)
        return X 

class ConvertObjToDateTime(BaseEstimator, TransformerMixin):
    def __init__(self, datetime_fields_info) -> None:
        super().__init__()
        self.datetime_fields_info = datetime_fields_info

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = dataprocessor.obj_to_datetime(X, self.datetime_fields_info)
        return X 

class ConvertObjToNumeric(BaseEstimator, TransformerMixin):
    def __init__(self, numeric_field_info) -> None:
        super().__init__()
        self.numeric_field_info = numeric_field_info

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = dataprocessor.numeric_conversion(X, self.numeric_field_info)
        return X 

class RemoveMissingValueInTargetVariable(BaseEstimator, TransformerMixin):
    def __init__(self, target_variable) -> None:
        super().__init__()
        self.target_variable = target_variable

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = dataprocessor.rm_rows_target_var(X, self.target_variable)
        return X 

class RemoveMissingValueInContinuousVariable(BaseEstimator, TransformerMixin):
    def __init__(self, continuous_variable) -> None:
        super().__init__()
        self.continuous_variable = continuous_variable

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X= dataprocessor.remove_missing(X, self.continuous_variable)
        return X 

class DataCleansing(BaseEstimator, TransformerMixin):
    def __init__(self, dirty_data_info) -> None:
        super().__init__()
        self.dirty_data_info = dirty_data_info

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = dataprocessor.dirty_data_processing(X, self.dirty_data_info)
        return X 

class DataValidation(BaseEstimator, TransformerMixin):
    def __init__(self, valid_data_info) -> None:
        super().__init__()
        self.valid_data_info = valid_data_info

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = dataprocessor.valid_data_processing(X, self.valid_data_info)
        return X 

class ImputeMissingValue(BaseEstimator, TransformerMixin):
    def __init__(self, impute_missing_val_info) -> None:
        super().__init__()
        self.impute_missing_val_info = impute_missing_val_info

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = dataprocessor.impute_missing_value_info(X, self.impute_missing_val_info)
        return X 

class LabelEncode(BaseEstimator, TransformerMixin):
    def __init__(self, non_numeric_cols) -> None:
        super().__init__()
        self.non_numeric_cols = non_numeric_cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = dataprocessor.label_encoder(X, self.non_numeric_cols)
        return X 
    
class DeriveYearFromDate(BaseEstimator, TransformerMixin):
    def __init__(self, derive_year_from_date) -> None:
        super().__init__()
        self.derive_year_from_date = derive_year_from_date

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = featureengineering.yyyy_from_date(X, self.derive_year_from_date)
        return X 

class OneHotEncoding(BaseEstimator, TransformerMixin):
    def __init__(self, one_hot_encode_fields) -> None:
        super().__init__()
        self.one_hot_encode_fields = one_hot_encode_fields

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = featureengineering.one_hot_key_encode(X, self.one_hot_encode_fields)
        return X 

class ConvertMilesToKM(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Convert None to NaN
        X = featureengineering.convert_miles_to_KM(X)
        return X  # Return the rows where null is changed to None

class CalYearDiff(BaseEstimator, TransformerMixin):
    def __init__(self, diff_years) -> None:
        super().__init__()
        self.diff_years = diff_years

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Convert None to NaN
        X = featureengineering.calc_year_diff(X, self.diff_years)
        return X  # Return the rows where null is changed to None

class PrepareData(BaseEstimator, TransformerMixin):
    def __init__(self, target_variable, test_size, random_state) -> None:
        super().__init__()
        self.target_variable = target_variable
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return modelbuild.prepare_data(X, self.target_variable, self.test_size, self.random_state)
    
def main() -> None:
    """
    Main function to demonstrate 
        1> Read configurations using the 'read_yaml' function
        2> Read pre_cruise 
        3> Read post_cruise data
        4> Merge Pre_cruise and Post_cruise to form df_cruise with Index as the key
        5> Create Data Cleansing Pipeline with the following functions:
            a> Convert NaN to Null
            b> Split composite field to seperate fields
            c> Remove columns with high missing threshold value
            d> Convert Object to DateTime
            e> Convert Object to Numeric 
            f> Remove missing value from target variable
            g> Remove missing value from Continuous Variables
            h> Remove known dirty data from variables
            g> Remove any unknown data from variables
            i> Impute missing values with values
            j> Perform label Encode on non numeric columns 
        6> Create Feature Engineering Pipeline with the following functions:
            a> Derive year from date
            b> Perform one hot key encode on multi-valued non numeric columns
            c> Standardise distance to KM
            d> Derive age from DOB
        7> Combine Data Cleansing Pipeline and Feature Engineering Pipeline into preprocessing 
            pipelines
        8> Execute the preprocessing using the df_cruise dataset
        9> Derive the selected model from yaml file and output accuracy score
    """
    # Read configurations using the 'read_yaml' function
    yaml_data = util.read_yaml(YAML_FILEPATHNAME)
    TEST_SIZE = yaml_data['test_size']
    RANDOM_STATE = yaml_data['random_state']
    TARGET_VARIABLE = yaml_data['target_variable']
    DB_INFO = yaml_data['databases']
    COMPOSITE_FIELD_INFO = yaml_data['composite_fields_to_split']
    ID_FIELDS = ast.literal_eval(yaml_data['ID_columns'])
    DATETIME_FIELD_INFO = yaml_data['convert_obj_datetime']
    NUMERIC_FIELD_INFO = yaml_data['convert_obj_numeric']
    MISSING_VAL_THRESHOLD =  yaml_data['pct_missing_threshold']
    CONTINUOUS_VARIABLE = ast.literal_eval(yaml_data['continuous_variables'])
    DIRTY_DATA_INFO = yaml_data['dirty_data_setting']
    VALID_DATA_INFO = yaml_data['valid_data_setting']
    NON_NUMERIC_COL = yaml_data['non_numeric_cols']
    DATE_YYYY_INFO = yaml_data['convert_date_yyyy']
    IMPUTE_MISSING_VALUE_INFO = yaml_data['impute_missing_value']
    OHE_FIELDS = ast.literal_eval(yaml_data['one_hot_encode'])
    DIFF_YEARS = yaml_data['diff_year']
    COL_TO_NORMALISE = ast.literal_eval(yaml_data['column_to_normalise'])
    SELECTED_MODEL = yaml_data['selected_model']
    SCALAR_OPT = yaml_data['scalar_option']

    # Read Pre_cruise data
    df_pre_cruise = database.db_read(DATA_PATH, DB_INFO[PRE_CRUISE_DB])

    # Read Post_cruise data
    df_post_cruise = database.db_read(DATA_PATH, DB_INFO[POST_CRUISE_DB])

    # Merge Pre_cruise and Post_cruise to form df_cruise with Index as the key
    df_cruise = database.db_merge_db (df_pre_cruise, df_post_cruise)
    print("Successfully load and merge data files.")

    print("Before PRINTOUT")
    print(df_cruise.info())

    # Data Cleaning Pipeline
    data_cleaning_pp = Pipeline(steps=[
        ('none_to_null', ConvertNanToNone()),
        ('split_composite_fields', SplitCompositeFields(COMPOSITE_FIELD_INFO)),
        ('remove_id_cols', RemoveIDsCols(ID_FIELDS)),
        ('missing_val_threshold', RemoveColumnWithHighMissingVal(MISSING_VAL_THRESHOLD)),
        ('obj_to_datetime', ConvertObjToDateTime(DATETIME_FIELD_INFO)),
        ('numeric_conversion', ConvertObjToNumeric(NUMERIC_FIELD_INFO)),
        ('rm_rows_target_var', RemoveMissingValueInTargetVariable(TARGET_VARIABLE)),
        ('remove_missing', RemoveMissingValueInContinuousVariable(CONTINUOUS_VARIABLE)),
        ('dirty_data_processing', DataCleansing(DIRTY_DATA_INFO)),
        ('valid_data_processing', DataValidation(VALID_DATA_INFO)),
        ('impute_missing_value_info', ImputeMissingValue(IMPUTE_MISSING_VALUE_INFO)),
        ('label_encode', LabelEncode(NON_NUMERIC_COL))
    ])

    # Feature Engineering Pipeline
    feature_engineering_pp = Pipeline(steps=[
        ('derive_year_from_date', DeriveYearFromDate(DATE_YYYY_INFO)),
        ('one_hot_encoding', OneHotEncoding(OHE_FIELDS)),
        ('convert_miles_to_KM', ConvertMilesToKM()),
        ('diff_date', CalYearDiff(DIFF_YEARS))
    ])

    # Combine data_cleaning_pp, feature_engineering_pp into cruise_pipeline
    cruise_pipeline = make_pipeline(data_cleaning_pp, feature_engineering_pp) 
    df_cruise = cruise_pipeline.transform(df_cruise)

    def fit_and_print(model, hyperparameters:dict, df_cruise:pd.DataFrame)-> None:
        """
        Fits a machine learning model, evaluates its performance on training and testing data,
        and prints the accuracy scores.

        Parameters:
            model: The machine learning model object.
            hyperparameters: Dictionary containing hyperparameters for the model.
            df_cruise: DataFrame containing the dataset.

        Returns:
            None
        """
        X_train, X_test, y_train, y_test = modelbuild.prepare_data(df_cruise, TARGET_VARIABLE,
                                                                   TEST_SIZE, RANDOM_STATE)
        X_train_smote, y_train_smote = modelbuild.SMOTE(X_train, y_train, RANDOM_STATE)
        X_train_normalised, X_test_normalised = modelbuild.normalised_data(X_train_smote, X_test, SCALAR_OPT, COL_TO_NORMALISE)
        y_train_pred, y_test_pred = model.model_processing(X_train_normalised, y_train_smote, X_test_normalised, hyperparameters)
        print("Train Accuracy Score : " + str(accuracy_score(y_train_smote, y_train_pred)))
        print("Test Accuracy Score : " + str(accuracy_score(y_test, y_test_pred)))
        return None

    # Initialised the selected model
    model = eval(SELECTED_MODEL['Model_Class_Name'])
    # Derived the hyperparameters values
    hyperparameters = ast.literal_eval(SELECTED_MODEL['Model_Hyperparameters'])
    # Build the selected model
    fit_and_print(model, hyperparameters, df_cruise)

if __name__ == "__main__":
    main()