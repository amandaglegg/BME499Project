# %%
# --- Importing Libraries ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import yellowbrick
import os
import pickle

from numpy.core.arrayprint import repr_format

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score, KFold

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import cross_val_score, KFold, GridSearchCV

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier

from matplotlib.collections import PathCollection
from statsmodels.graphics.gofplots import qqplot
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
from yellowbrick.classifier import PrecisionRecallCurve, ROCAUC, ConfusionMatrix
from yellowbrick.style import set_palette
from yellowbrick.model_selection import LearningCurve, FeatureImportances
from yellowbrick.contrib.wrapper import wrap

# --- Libraries Settings ---
warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.rcParams['figure.dpi']=100
set_palette('dark')

from numpy import (

    mean, absolute

)


#%% 
# --- Hypertuning Function for Random Forest ---

def RfhyperParameterTuning(X_train, y_train):


    param_grid = {

    'bootstrap': [True],

    'max_depth': [1,3,5, 10, 40, 70, 100, None], # max number of levels in each decision tree

    'max_features': [2, 3, 'auto', 'sqrt'], # max number of features considered for splitting a node

    'min_samples_leaf': [1, 2, 3, 4, 5], # min number of data points allowed in a leaf node

    'min_samples_split': [2, 4, 8, 10, 12], #  min number of data points placed in a node before the node is split

    'n_estimators': [5, 50, 200, 400, 800, 1000, 1400, 3000] # number of trees in the forest

    }

    rf = RandomForestRegressor()

    gsearch = GridSearchCV(estimator = rf,

                           param_grid = param_grid,                       

                           cv = 5,

                           n_jobs = -1,

                           verbose = 2)

    gsearch.fit(X_train,y_train)

    return gsearch.best_params_

#%% 
# --- Hypertuning Function for Extra Trees ---

def EThyperParameterTuning(X_train, y_train):

     param_tuning = {

        'min_samples_split': [2, 5, 7, 10, 14],

        'max_features': [1, 2, 'sqrt', 4, 7, 11],

        'n_estimators': [5, 50, 200, 400, 800, 1000, 1400, 3000],

        'min_samples_leaf': [2, 5, 10, 20, 30, 50],

        'max_depth': [1,3,5, 10, 40, 70, 100, None],

     }


     ET = ExtraTreesClassifier()

     gsearch = GridSearchCV(estimator = ET,

                            param_grid = param_tuning,                       

                            cv = 5,

                            n_jobs = -1,

                            verbose = 2)

     gsearch.fit(X_train,y_train)

     return gsearch.best_params_

#%% 
# --- Hypertuning Function for Gradient Boosting ---

def GBhyperParameterTuning(X_train, y_train):

    param_tuning2= {

       'max_depth': [1,3,5, 10, 40, 70, 100, None],

       'min_samples_leaf': [2, 5, 10, 20, 30, 50],

       'min_samples_split': [2, 5, 7, 10, 14],

       'max_features': [1, 2, 4, 8, 'sqrt'],

       'learning_rate': [0.01, 0.1, 1, 10, 100],

       'n_estimators': [5, 50, 200, 400, 800, 1000, 1400, 3000],

       'subsample': [0.5, 0.7, 0.8],

       'random_state': ['None', 0, 42]
       
    }

    GB = GradientBoostingClassifier()

    gsearch = GridSearchCV(estimator = GB,

                           param_grid = param_tuning2,                       

                           cv = 5,

                           n_jobs = -1,

                           verbose = 2)

    gsearch.fit(X_train,y_train)

    return gsearch.best_params_

# %% 
# --- Importing Dataset ---

os.chdir("..") #move up one directory to BME 499
our_path = os.path.abspath(os.curdir)
our_path = our_path + '/datasets/hch.csv'
df = pd.read_csv(our_path)
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_', regex=True)

#%%
# --- Drop Data Columns ---

df = df.drop(columns=['cholesterol', 'fasting_blood_sugar', 'resting_ecg'])

#%%
# remapping variable names to correct those from previous data set
df['st_slope'].value_counts()
#remap_dict = {1: 'Up', 2: 'Flat', 3: 'Down'}
#df['st_slope'] = df['st_slope'].replace(remap_dict)
df = df.loc[df['st_slope'] != 0].copy() 

# %%
# --- Seperating Dependent Features ---
x = df.drop(['target'], axis=1)
y = df['target']

# --- Data Normalization using Min-Max Method ---
x = MinMaxScaler().fit_transform(x)

# --- Splitting Dataset into 80:20 ---
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)


#%%
# --- running random forest function
# rf_params = RfhyperParameterTuning(x, y)
# print(f'These are them params {rf_params}')
# %%
# --- running gradient boosting function
#rf_params = GBhyperParameterTuning(x, y)
#print(f'These are them GBparams {rf_params}')

# %%
# --- running extra trees function
rf_params = EThyperParameterTuning(x, y)
print(f'These are them GBparams {rf_params}')
