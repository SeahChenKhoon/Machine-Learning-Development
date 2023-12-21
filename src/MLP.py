from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

import ast

import util
import database
from data_preprocessing import DataProcessing

DATA_PATH =  "../data/"
YAML_FILEPATHNAME = "../config.yaml"
PRE_CRUISE_DB = 0
POST_CRUISE_DB = 1
IS_NOTEBOOK = True

dataprocessor = DataProcessing()

class ConvertNanToNone(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Convert None to NaN
        X = dataprocessor.replace_nan_none(X)
        return X  # Return the rows where null is changed to None
    
class SplitCompositeFields(BaseEstimator, TransformerMixin):
    def __init__(self, composite_fields_info) -> None:
        super().__init__()
        self.composite_fields_info = composite_fields_info

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        dataprocessor.split_composite_field(X, self.composite_fields_info)
        return X  

class RemoveColumnWithHighMissingVal(BaseEstimator, TransformerMixin):
    def __init__(self, missing_val_threshold) -> None:
        super().__init__()
        self.missing_val_threshold = missing_val_threshold
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        dataprocessor.rm_cols_high_missing(X, self.missing_val_threshold)
        return X  

class RemoveIDsCols(BaseEstimator, TransformerMixin):
    def __init__(self, id_cols) -> None:
        super().__init__()
        self.id_cols = id_cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        dataprocessor.rm_id_cols(X, self.id_cols)
        return X 

class ConvertObjToDateTime(BaseEstimator, TransformerMixin):
    def __init__(self, datetime_fields_info) -> None:
        super().__init__()
        self.datetime_fields_info = datetime_fields_info

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        dataprocessor.obj_to_datetime(X, self.datetime_fields_info)
        return X 

class ConvertObjToNumeric(BaseEstimator, TransformerMixin):
    def __init__(self, numeric_field_info) -> None:
        super().__init__()
        self.numeric_field_info = numeric_field_info

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        dataprocessor.numeric_conversion(X, self.numeric_field_info)
        return X 

# ('numeric_conversion', ConvertObjToNumeric(NUMERIC_FIELD_INFO))
# dp.numeric_conversion(df_cruise, NUMERIC_FIELD_INFO)


def main():
    # Read YAML file
    yaml_data = util.read_yaml(YAML_FILEPATHNAME)
    DISPLAY_STUB = yaml_data['display_stub']
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
    VERBOSE = yaml_data['verbose']
    LR_HYPERPARAM = yaml_data['hyperparameters']['lr_param']
    DTC_HYPERPARAM = yaml_data['hyperparameters']['dtc_param']
    RFC_HYPERPARAM = yaml_data['hyperparameters']['rfc_param']
    GBC_HYPERPARAM = yaml_data['hyperparameters']['gbc_param']

    # Read Pre_cruise data
    df_pre_cruise = database.db_read(DATA_PATH, DB_INFO[PRE_CRUISE_DB])

    # Read Post_cruise data
    df_post_cruise = database.db_read(DATA_PATH, DB_INFO[POST_CRUISE_DB])

    # Merge Pre_cruise and Post_cruise to form df_cruise with Index as the key
    df_cruise = database.db_merge_db (df_pre_cruise, df_post_cruise)
    print("Successfully load and merge data files.")

    # Data Cleaning Pipeline
    data_cleaning_pipeline = Pipeline(steps=[
        ('none_to_null', ConvertNanToNone()),
        ('split_composite_fields', SplitCompositeFields(COMPOSITE_FIELD_INFO)),
        ('remove_id_cols', RemoveIDsCols(ID_FIELDS)),
        ('missing_val_threshold', RemoveColumnWithHighMissingVal(MISSING_VAL_THRESHOLD)),
        ('obj_to_datetime', ConvertObjToDateTime(DATETIME_FIELD_INFO)),
        ('numeric_conversion', ConvertObjToNumeric(NUMERIC_FIELD_INFO))
    ])
    
    df_cruise = data_cleaning_pipeline.transform(df_cruise)
    print("FINAL PRINTOUT")
    print(df_cruise.info())

if __name__ == "__main__":
    main()