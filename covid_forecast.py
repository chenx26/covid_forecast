# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.2
#   kernelspec:
#     display_name: covid_forecast
#     language: python
#     name: covid_forecast
# ---

# +
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import jupyternotify
ip = get_ipython()
ip.register_magics(jupyternotify.JupyterNotifyMagics)
# -

url_to_covid = 'https://covid.ourworldindata.org/data/owid-covid-data.csv'

df_orig = pd.read_csv(url_to_covid)

df_orig.shape

df_orig.columns

df_orig.date

df_orig.location.unique()

df = df_orig[df_orig.location == 'United States']

df.shape

df = df.sort_values('date')

plt.scatter(df.date, df.new_cases)

df = df_orig.copy()

df.info()

df.shape

# +

percent_missing = df.isnull().sum() * 100 / len(df)
missing_value_df = pd.DataFrame({'column_name': df.columns,
                                 'percent_missing': percent_missing})

missing_value_df.sort_values('percent_missing', inplace=True, ascending=False)
missing_value_df
# -

cols_too_many_missing = missing_value_df[missing_value_df.percent_missing > 50].index.tolist()
len(cols_too_many_missing)
cols_too_many_missing

df_orig.head()

target_var = 'new_cases'
df = df_orig.dropna(subset=[target_var])

from sklearn.preprocessing import OneHotEncoder

df.shape

df.dtypes

df.tests_units.unique()

df_cat = df.select_dtypes('object')

df_fp = df.select_dtypes('float64')

df_fp.shape



df_cat = pd.get_dummies(df_cat, dummy_na=True)

df_cat.shape

df_cat.head()

y = df_fp[target_var]

df_fp = df_fp[[x for x in df_fp.columns if x!=target_var]]

df_fp.shape

X = pd.concat([df_fp, df_cat], axis=1)

import xgboost as xgb

reg = xgb.XGBRegressor(n_estimators = 10)

reg.fit(X, y)

np.unique(reg.predict(X))

# +
# reg.score?
# -

y.shape

X.head()

X.describe()

X.head(20)

X.total_cases.tail()

y.isnull().sum()


