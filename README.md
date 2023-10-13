# SEAH-CHEN-KHOON-chenkhoon@yahoo.com
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
* Download the source from https://github.com/SeahChenKhoon/Machine-Learning-Development to a local drive
* Ensure that the cruise_post.db and cruise_pre.db are copied to the data folder 
* The requirement of this project can be found in the reference folder (AIAP Batch 15 Technical Assessment v2.pdf)

### Executing program
* Launch Git Bash
* Redirect to the downloaded folder. 
* Type in "sh run.sh"
* This action will 
    * install the required libraries found in requirements.txt
    * execute the main.py
* Execution of the main.py will perform
    * data_preprocessing.py
        * Read the data
        * Read config file (config.yaml)
        * Data Cleansing for
            * Date of Birth
            * Compute Age from DOB
            * Mileage Conversion
            * Remove missing rows from target variable
            * Resolve invalid data found in various feature columns
        * Perform encoding
            * Label Encoding
            * Ordinal Encoding
            * One hot key Encoding
        * Impute Missing Data
        * Remove duplicated rows
        * Remove outliers
        * Drop unnecessary columns
        * Data Splitting of independent and dependent variables
        * Perform standard scaler
    * feature_engineering.py
        * Group various independent variables in one 
    * model_build.py
        * For each of the model:
            * Train the model
            * Predict the train dataset
            * Predict the test dataset
    * model_eval.py
        * Output the accuracy of predictions based on
            * accuracy_score
            * f1_score
            * precision_score
            * recall_score

## Authors
Contributors names and contact info
    Seah Chen Khoon. Email: chenkhoon@yahoo.com  

