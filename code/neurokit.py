#%%
# *****Note: you have to run the cells (interactively) to see graphs
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

# --- Process dataset: normalize between 0 and 1 and convert to one array ---
def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))
df_norm = NormalizeData(df)
print("normalized df", df_norm)
df = df_norm.iloc[:,0].to_numpy() 
print("converted to array", df)

# --- Calculate sampling freq and period
real_freq = len(df)/30
period = 1/ real_freq

# Similate fake data for comparison
# Generate 15 seconds of ECG signal (recorded at 250 samples / second)
ecg = nk.ecg_simulate(duration=15, sampling_rate=250, heart_rate=70)

# Process it
signals_real, info = nk.ecg_process(df, sampling_rate = real_freq)
signals_fake, info = nk.ecg_process(ecg, sampling_rate = 250)
# Visualise the processing
print("real plot")
nk.ecg_plot(signals_real, sampling_rate=real_freq)
print("fake plot")
nk.ecg_plot(signals_fake, sampling_rate=250)


# Download data
# ecg_signal = nk.data(dataset="ecg_3000hz")
ecg_signal_real = df
ecg_signal_fake = ecg

# Extract R-peaks locations
_, rpeaks1 = nk.ecg_peaks(ecg_signal_real, sampling_rate= real_freq)
_, rpeaks2 = nk.ecg_peaks(ecg_signal_fake, sampling_rate= 250)
# Delineate
signal, waves = nk.ecg_delineate(ecg_signal_real, rpeaks1, sampling_rate= real_freq, method="dwt", show=True, show_type='all')
signal, waves = nk.ecg_delineate(ecg_signal_fake, rpeaks2, sampling_rate= 250, method="dwt", show=True, show_type='all')

