#processing ECG for T-wave abnormality or Estes' criteria.  
#Requires multiple ECG measurements with apple watch to simulate multi-lead measurements
#from Heart disease datset, classification is 0=normal 1=ST-T abnormality (T wave inversion OR ST elevation or depression >0.05mV), 
# 2= 4 or more 4pts of Estes criteria.

# Load NeuroKit and other useful packages
import neurokit2 as nk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn import preprocessing
#%matplotlib inline
plt.rcParams['figure.figsize'] = [8, 5]  # Bigger images

# --- Importing Dataset ---
# df = pd.read_csv("D:/Documents/4B/BME499/github/BME499Project/datasets/ecg_2020-06-01.csv", header=9, usecols = ['Unit'])
# Import from virtual path, make sure ur in the right folder when u run the code
os.chdir("..") #move up one directory to BME 499
our_path = os.path.abspath(os.curdir)
our_path = our_path + '/datasets/ecg_2020-06-01.csv'
print(our_path) # check the path is correct

df = pd.read_csv(our_path, header=9, usecols = ['Unit'])
print("original df",df)

# --- Process dataset: convert uV to mV, divide by 1000, and convert to array---
df = df/1000
df = df.iloc[:,0].to_numpy()
print("converted to mV", df)

# --- Calculate sampling freq and period
real_freq = len(df)/30
period = 1/ real_freq