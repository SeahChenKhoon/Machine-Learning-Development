import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def vs_confusion_matrix(prt_label, y_actual, y_predict, is_notebook):
    # print(y_actual.shape)
    # print(y_predict.shape)
    print("\033[1m" + prt_label + "\033[0m")
    if is_notebook == True:
        cm=confusion_matrix(y_actual, y_predict, labels=[0, 1])
        df_cm = pd.DataFrame(cm, index = [i for i in ["Actual Blastoise","Actual Lapras"]],
            columns = [i for i in ["Predicted Blastoise","Predicted Lapras"]])
        plt.figure(figsize = (7,5))
        sns.heatmap(df_cm, annot=True, fmt="d",cmap='Blues')
        plt.show()
    else:
        print(confusion_matrix(y_actual, y_predict))


def vs_countplot_both(df_dataframe: pd.DataFrame, x_col:str, hue_col:str=None)->None:
    plt.figure(figsize=(10, 4))
    ax = sns.countplot(x=x_col,  data=df_dataframe)
    for container in ax.containers:
        ax.bar_label(container)
    plt.show()

    plt.figure(figsize=(10, 4))
    ax = sns.countplot(x=x_col,  data=df_dataframe, hue='Ticket Type')
    for container in ax.containers:
        ax.bar_label(container)
    plt.show()
    return None

def vs_countplot(df_dataframe: pd.DataFrame,  x_col:list[str]=None, display_count:bool=False, hue_col:str=None)->None:
    if x_col != None:
        for col in x_col:
            print(col)
            plot_countplot(df_dataframe, col, display_count)
    return None

def plot_countplot(df_dataframe, col_name, display_count):
    plt.figure(figsize=(8, 4))
    ax = sns.countplot(x=col_name,  data=df_dataframe)
    if display_count == True:
        for container in ax.containers:
            ax.bar_label(container)
    plt.show()
    return None

def vs_countplot_target(df_dataframe: pd.DataFrame, x_col:str, hue_col:str=None)->None:
    plt.figure(figsize=(4, 4))
    ax = sns.countplot(x=x_col,  data=df_dataframe, hue='Ticket Type')
    for container in ax.containers:
        ax.bar_label(container)
    plt.show()
    return None

def vs_pieplot(dataset, col_name):
    dataset = pd.DataFrame(dataset)
    # Calculate the value counts and percentages
    value_counts = dataset[col_name].value_counts()
    percentages = (value_counts / len(dataset)) * 100
    # Create a pie plot
    plt.figure(figsize=(4, 4))
    plt.pie(percentages, labels=percentages.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('pastel'))
    # Add a title
    plt.title('Distribution of ' + col_name)
    plt.show()

def vs_plot_corr_chart(df_dataframe: pd.DataFrame)->None:
    corr_matrix = df_dataframe.corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    # Plot the heatmap with the diagonal elements hidden
    plt.figure(figsize=(20, 20))
    sns.heatmap(corr_matrix, annot=True, mask=mask, vmin=-1, vmax=1, cmap="coolwarm")
    plt.title('Correlation Coefficient Of Predictors')
    plt.show()