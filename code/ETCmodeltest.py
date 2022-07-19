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

from matplotlib.collections import PathCollection
from statsmodels.graphics.gofplots import qqplot
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
from yellowbrick.classifier import PrecisionRecallCurve, ROCAUC, ConfusionMatrix
from yellowbrick.style import set_palette
from yellowbrick.model_selection import LearningCurve, FeatureImportances
from yellowbrick.contrib.wrapper import wrap
import pickle
# --- Libraries Settings ---
warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.rcParams['figure.dpi']=100
set_palette('dark')

# %% 
# --- Importing Dataset ---
#df = pd.read_csv("D:/Documents/4B/BME499/test/BME499Project/datasets/heart.csv")

os.chdir("..") #move up one directory to BME 499
our_path = os.path.abspath(os.curdir)
user_path = our_path + '/datasets/fake_user_data.csv'
our_path = our_path + '/datasets/hch.csv'
print(our_path)

df = pd.read_csv(our_path)
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_', regex=True)
# --- Reading Dataset ---
df.head().style.background_gradient(cmap='Reds').set_properties(**{'font-family': 'Segoe UI'}).hide_index() 

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
# x = MinMaxScaler().fit_transform(x)

# --- Splitting Dataset into 80:20 ---
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)

infile = open('ETC_model_not_normalized.pkl','rb')
model = pickle.load(infile)
infile.close()

#%% Test code
# Find the code that takes the test values and runs them through
# the model and gives you the accuracy table back

y_pred_ET = model.predict(x_test)

# --- ET Accuracy ---
ETAcc = accuracy_score(y_pred_ET, y_test)
percentage = ETAcc*100
print("The percentage accuracy of the model is:", percentage)

