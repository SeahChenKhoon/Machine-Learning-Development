# aiap15-SEAH-CHEN-KHOON-S7134252B
AIAP15 Techical Assessment

# Project Title
Prediction of ticket type that potential customers will purchase.

## Description
Based on a list of attributes found in pre and post survey, this system ought to provide prediction on ticket type purchase

## Getting Started

### Dependencies

* For windows system, 
    * need to install ubuntu
    * In "Turn Windows features on or off", check on "Virutal Machine Platform" and "Windows Subssystem for Linux"


### Installing
* Installing of git bash
* Download the source from https://github.com/SeahChenKhoon/aiap15-SEAH-CHEN-KHOON-S7134252B to a local drive
* Ensure that the cruise_post.db and cruise_pre.db are copied to the data folder under aiap15-SEAH-CHEN-KHOON-S7134252B

### Executing program
* Launch Git Bash
* Redirect to aiap15-SEAH-CHEN-KHOON-S7134252B folder
* Type in "sh run.sh"
* This action will 
    * install the required libraries found in requirements.txt
    * execute the main.py
* Execution of the program
    * EDA (eda.py) which includes
        * Read Data. Read of 2 MD files into dataframes and combine them into one dataframes.
        * Standardise Cruise Distance to KM. This feature contains miles and km so standardise to km.
        * Convert DOB to Datetime by removing non-datetime records
        * Fix typo error in Cruise Name
    * Covert non-numeric data field to numeric (feature_engineering.py) which is to  
        * Convert columns with binary values to numeric 
            * Gender - Female to 0 and Male to 1 
            * Cruise Name - Blastoise to 0 and Lapras to 1
        * Transform DOB to another new column Age
        * Perform Ordinal Encoding on 
            * Onboard Wifi Service
            * Onboard Dining Service
            * Onboard Entertainment
        * Perform Label Encode on "Ticket Type"
        * Perform One hot Key Encode on "Source of Traffic"
    * Impute missing values (feature_engineering.py) which include:
        * Impute the following to mode
            * "Onboard Wifi Service","Embarkation/Disembarkation time convenient", 
                              "Ease of Online booking", "Gate location","Onboard Dining Service","Online Check-in","Cabin Comfort",
                              "Onboard Entertainment","Cabin service","Baggage handling","Port Check-in Service","Onboard Service",
                              "Cleanliness","Cruise Name"
        * Impute Mean to "Distance in KM"
        * Impute median to Age
        * Impute random to Gender 
    * Train Model (model.py)
        * Change the feature list in feature_list
        * Implemented decision_tree, knn and random_forest.
            * decision_tree.
                * Model supports multiclass classification
                * Model can support model complex relationships in the data
            * KNN
                * Simple and effective for small to medium-sized datasets.
                * Suitable for multiclass classification
            * Random Tree
                * Usage of multiple decision trees so less prone to overfitting compared to individual decision trees
                * Good for both binary and multiclass classification
        * The fine-tuning can be configured in the dictionary specified in model_params. This will allow user to add or remove model easily.
        * Note that I have done the fine-tuning based on divide and conquer that is by choosing some features one at a time to execute and see the respective mean_test_score to decide which to configure to. A summary is listed at the bottom of each hyperparameter model. Each of the findings will have a excel at the end of the statement which I will include in model.zip.
        * The best score and best parameter for each run can be found data\df_best_score_*.csv. This file can be used to determine the more effective algorithm.
        * Lastly the program will output the final dataset in data\df_cruise_final_*.csv for each run.

## Help
* Should you encounter any error in retriving the data, you may use Notepad to open "util.py" under src folder and amend the data path accordinly.
    data_path="..\data\\"
* My contact details can be found in the Author section. Please contact me should you find any issue. Thanks in advance.

## Authors
Contributors names and contact info
    Seah Chen Khoon. Tel: 97868397  

## Acknowledgments
* I like to express great thanks to AI Singapore for giving me the chance to participate in this Techical Assessment. 
* Special thanks to mentors Kelvin Chong, Meldrick and Benjamin for providing guidance and advice during the bootcamp. 
* With their advice, I implemented the following:
    * Read db file to dataframes using util function
    * Implemented doc stream and data type links on functions
    * Point out that I it is wrong convert my ipynb file to py by exporting.
    * I learn the use of different encoding methods (label encode, one hot key, ordinal encode) correctly.
    * run.sh
    * AND many, many more......  