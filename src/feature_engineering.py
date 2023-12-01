import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def plot_corr_chart(df_dataframe: pd.DataFrame)->None:
    corr_matrix = df_dataframe.corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    # Plot the heatmap with the diagonal elements hidden
    plt.figure(figsize=(20, 20))
    sns.heatmap(corr_matrix, annot=True, mask=mask, vmin=-1, vmax=1, cmap="coolwarm")
    plt.title('Correlation Coefficient Of Predictors')
    plt.show()

# class FeatureEngineer:
#     def feature_grouping(self, merged_data:pd.DataFrame, column_list:list[str], column_grp:str)->None:
#         """
#         Perform feature grouping on the input DataFrame.

#         Parameters:
#             merged_data (pd.DataFrame): The input DataFrame.
#             column_list (list[str]): List of column names to calculate the mean from.
#             column_grp (str): Name of the new column to be added.

#         Returns:
#             None
#         """
#         mean_val = merged_data[column_list].mean(axis=1).round()
#         merged_data[column_grp] = mean_val
#         util.drop_columns(merged_data, column_list)
#         return
