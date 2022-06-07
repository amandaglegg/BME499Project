#Plot raw ECG#
# by Chris and Christine#

#Import libraries#
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import neurokit2 as nk
import seaborn as sns

plt.rcParams['figure.figsize'] = [8, 5]  # Bigger images

# --- Importing Dataset ---
# df = pd.read_csv("D:/Documents/4B/BME499/github/BME499Project/datasets/ecg_2020-06-01.csv", header=9, usecols = ['Unit'])
os.chdir("..") #move up one directory to BME 499
our_path = os.path.abspath(os.curdir)
our_path = our_path + '/datasets/ecg_2020-06-01.csv'
print(our_path)
df = pd.read_csv(our_path, header=9, usecols = ['Unit'])

real_freq = len(df)/30
period = 1/ real_freq
#plot original ECG#
x = np.arange(0, 30, period)
y = df
plt.plot(x, y.to_numpy())
# naming the x axis
plt.xlabel('Time(s)')
# naming the y axis
plt.ylabel('microVolts (uV)')
# giving a title to my graph
plt.title('raw ECG graph')
# function to show the plot
plt.show()


