import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def vs_countplot_both(df_dataframe: pd.DataFrame, x_col:str, hue_col:str=None)->None:
    plt.figure(figsize=(4, 4))
    ax = sns.countplot(x=x_col,  data=df_dataframe)
    for container in ax.containers:
        ax.bar_label(container)
    plt.show()

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