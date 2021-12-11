"""
In Jupyter Notebook:
from base_code.explore import univar

univar(df,col,"relativa"); # <- "absoluta" "relativa" "acumulada"
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# Find the row number for a given column value.
def find_row_number(df,column,filter_value):
    return df.index[df[column] == filter_value][0]

# Get a record from a pandas dataframe given the row number.
def get_record(df,row_position):
    return df.loc[row_position]


def remove_outliers(dataframe,column):
    q3, q1 = np.percentile(dataframe[column],[75,25])
    iqr = q3-q1
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)
    dataframe = dataframe[(dataframe[column] >= lower_bound) & (dataframe[column] <= upper_bound)]
    return dataframe
    
def find_lower_outliers(df,col):
    q1 = df[col].describe().values[4]
    q3 = df[col].describe().values[6]
    iqr = q3-q1
    lower_bound = q1 - (1.5 * iqr)
    return set([val for val in list(df[col].where(df[col] < lower_bound).values) if str(val) != "nan"])

def find_upper_outliers(df,col):
    q3, q1 = np.percentile(df[col],[75,25])
    iqr = q3-q1
    upper_bound = q3 + (1.5 * iqr)
    return set(df.loc[df[col] > upper_bound,col].values)

def df_trans(in_df,ht,var):
    out_df = pd.DataFrame()
    c=0
    for gen in in_df[var]:
        for k,v in ht.items():
            out_df.loc[c,"generacion"] = gen
            out_df.loc[c,"medio_compra"] = v
            #out_df.loc[c,"total_compras"] = in_df.loc[in_df.index[in_df[var]==gen][0],k]
            out_df.loc[c,"total_compras"] = get_record(in_df,find_row_number(in_df,var,gen))
            c+=1
    return out_df
