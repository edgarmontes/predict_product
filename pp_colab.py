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

# Statistics
import scipy.stats as st
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Custom Libraries
from base_code.describe_df import data_description
from base_code.explore import remove_outliers,find_upper_outliers, find_lower_outliers,get_record#, create_boxplot
#from base_code.feature_selection import variable_normalization,select_kbest
from base_code.modeling import manual_model,final_model,linear_regression,random_forest_regressor

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
#
df = pd.read_csv("data/marketing_campaign.csv","\t")
#
df.columns = [re.sub(r"(?<=[a-z])(?=[A-Z])","_",col).lower() if col.find("_") == -1 else col.lower() for col in df.columns]
#
df.dropna(inplace=True)
#
df.drop(columns = ['z_costcontact', 'z_revenue'], inplace = True)
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
cols_traditional_model = ["mnt_wines","income","client_age","total_children"]
df_traditional_model = df[cols_traditional_model]
x = df_traditional_model[cols_traditional_model[1:]]
y = df_traditional_model[cols_traditional_model[0]]
#
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 2)
#
x_train_final, x_validation, y_train_final, y_validation = train_test_split(x_train, y_train, test_size = 0.2, random_state = 3)
#
multi_model = Pipeline(steps=[("Scaler",StandardScaler()),("model",LinearRegression())])
#
param_grid_1 = [{"model":[LinearRegression()],
               "model__fit_intercept":[True,False],
               "model__copy_X":[True]},
              {"model":[RandomForestRegressor()],
               "model__max_depth":[1,2,3,4,5,6],
               "model__n_estimators":[100,200,300,400,500],
               "model__min_samples_split":[2,3,4,5,6],
               "model__max_samples":np.linspace(0.01,1,10),
               "model__bootstrap":[True]},
              {"model":[KNeighborsRegressor()],
               "model__n_neighbors":[1,2,3,4,5,6],
               "model__algorithm":["auto","ball_tree","kd_tree","brute"],
               "model__leaf_size":[30,60,90]},
              {"model":[DecisionTreeRegressor()],
               "model__criterion":["squared_error","absolute_error"],
               "model__splitter":["best","random"],
               "model__max_depth":[1,2,3,4,5,6]}]
#
param_grid_2 = [{"model":[RandomForestRegressor()],
                 "model__max_depth":[1,2,3]},
                {"model":[KNeighborsRegressor()],
                 "model__n_neighbors":[1,2,3],
                 "model__algorithm":["auto","ball_tree","kd_tree","brute"]},
                {"model":[DecisionTreeRegressor()],
                 "model__criterion":["squared_error","absolute_error"],
                 "model__splitter":["best","random"]}]
#
grid_search_cv = GridSearchCV(estimator=multi_model,param_grid=param_grid_2,cv=5,scoring="neg_root_mean_squared_error")
#
grid_search_cv.fit(x_train,y_train)
#
results = pd.DataFrame(grid_search_cv.cv_results_).sort_values(by='rank_test_score')
print("\n",results.shape)
print("\n",results.head(12))
print("\n",results.tail(12))
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
print("\n",x_test.sort_values(by=["abs_percent"],ascending=True).head(60))
#
print("\n",x_test.sort_values(by=["abs_percent"],ascending=False).head(60))

