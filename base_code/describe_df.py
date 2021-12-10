"""
In Jupyter Notebook:
from base_code.describe_df import data_description

df = data_description("data_sources/marketing_campaign.csv","\t")
"""
import pandas as pd
from IPython.display import display,HTML
import seaborn as sns
import matplotlib.pyplot as plt
import re 


def data_description(input_file,delimiter):
    input_df = pd.read_csv(input_file,sep=delimiter,low_memory=False)
    
    print("\n****** Rename Columns: ******")
    print("\nOriginal column names:\n",input_df.columns)
    input_df.columns = [re.sub(r"(?<=[a-z])(?=[A-Z])","_",col).lower() if col.find("_") == -1 else col.lower() for col in input_df.columns]
    print("\nRenamed column names:\n",input_df.columns)
    
    print("\n****** Data Frame Shape: ******\n")
    print(input_df.shape)
    
    print("\n****** Head and Tail: ******\n")
    display(HTML(input_df.head(12).to_html()))
    display(HTML(input_df.tail(12).to_html()))
    
    print("\n****** Describe: ******\n")
    display(HTML(input_df.describe(include="object").to_html()))
    display(HTML(input_df.describe(exclude="object").to_html()))

    print("\n****** Nulls by column: ******\n")
    print(input_df.isnull().sum())
    
    print("\n****** Info: ******\n")
    input_df.info()
    
    print("\n****** Duplicated records: ******\n")
    filter_dups = input_df.duplicated()
    display(HTML(input_df[filter_dups].to_html()))
    
    print("\n****** Columns correlation: ******\n")
    display(HTML(input_df.corr().to_html()))
    
    print("\n****** Total unique values by column: ******\n")
    records = input_df.shape[0]
    for col in input_df.columns:
        categories_by_col = input_df[col].nunique()
        percent = round((categories_by_col/records)*100,2)
        print(f"Number of unique values for column {col}: ",categories_by_col,f"| Percent = {percent}%")

    print("\n****** Count for unique values by column(v2): ******\n")
    for col in input_df.columns:
        print(input_df[col].value_counts())
        print("\n")

    return input_df
