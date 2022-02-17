# Data Management
import pandas as pd
import numpy as np
import datetime
import scipy.stats as st
import re

# Plots
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use(["ggplot","seaborn-white"])
#plt.style.use(['fivethirtyeight','seaborn-poster'])
sns.set_style('whitegrid')
from IPython.display import display,HTML

# Libreries for modeling
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import ShuffleSplit
import sklearn.preprocessing
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest,f_regression
from sklearn import metrics

# Molde selection
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV


#=======================================================
def remove_outliers(dataframe,column):
    q3, q1 = np.percentile(dataframe[column],[75,25])
    iqr = q3-q1
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)
    dataframe = dataframe.sort_values(by=[column])
    dataframe = dataframe[(dataframe[column] >= lower_bound) & (dataframe[column] <= upper_bound)]
    return dataframe
#=======================================================
def multi_mod(df,param_grid,cols_traditional_model = ["mnt_wines","income","client_age","total_children"]):
    #
    ###df = pd.read_csv("data/marketing_campaign.csv","\t")
    #
    df.columns = [re.sub(r"(?<=[a-z])(?=[A-Z])","_",col).lower() if col.find("_") == -1 else col.lower() for col in df.columns]
    #
    df.dropna(inplace=True)
    #
    #df.drop(columns = ['z_costcontact', 'z_revenue'], inplace = True)
    #
    current_date = datetime.datetime.now()
    date = current_date.date()
    df["client_age"] = date.year - df["year_birth"]
    #
    df = remove_outliers(df,"income")
    #
    df = remove_outliers(df,"mnt_wines")
    #
    df["total_children"] = df["kidhome"] + df["teenhome"]
    #
    #cols_traditional_model = ["mnt_wines","income","client_age","total_children"]
    df_traditional_model = df[cols_traditional_model]
    x = df_traditional_model[cols_traditional_model[1:]]
    y = df_traditional_model[cols_traditional_model[0]]
    #
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 2)
    #
    multi_model = Pipeline(steps=[("Scaler",StandardScaler()),("model",LinearRegression())])
    #
    grid_search_cv = GridSearchCV(estimator=multi_model,param_grid=param_grid,cv=5,scoring="neg_root_mean_squared_error")
    #
    grid_search_cv.fit(x_train,y_train)
    #
    results = pd.DataFrame(grid_search_cv.cv_results_).sort_values(by="rank_test_score")
    print("\n",results.shape)
    display(HTML(results.head(12).to_html()))
    display(HTML(results.tail(12).to_html()))
    #
    print("\nBest score: ",grid_search_cv.best_score_)
    #
    print("\n",grid_search_cv.best_params_)
    #
    print("\n",grid_search_cv.best_estimator_)
    #
    grid_search_cv.fit(x_train,y_train)
    #
    y_pred = grid_search_cv.predict(x_test)
    #
    x_test["y_test"] = y_test
    x_test["y_pred"] = y_pred
    x_test["residuals"] = x_test["y_test"] - x_test["y_pred"]
    x_test["abs_percent"] = (round(x_test["residuals"]/x_test["y_test"],6)*100).abs()
    #
    display(HTML(x_test.sort_values(by=["abs_percent"],ascending=True).head(60).to_html()))
    #
    display(HTML(x_test.sort_values(by=["abs_percent"],ascending=False).head(60).to_html()))
