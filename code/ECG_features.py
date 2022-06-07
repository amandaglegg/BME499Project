#Extract Features from ECG#
# by Chris and Christine#

#Import libraries#
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Importing Dataset ---
df = pd.read_csv("D:/Documents/4B/BME499/github/BME499Project/datasets/ecg_2020-06-01.csv", header=9, usecols = ['Unit'])

sampling_freq = df.iloc[6,0]
#period = 1/sampling_freq

#plot#
x = np.arange(0, 30, 0.0019468)
y = df
plt.plot(x, y)
  
# naming the x axis
plt.xlabel('Time(s)')
# naming the y axis
plt.ylabel('microVolts (uV)')
  
# giving a title to my graph
plt.title('ECG graph')
  
# function to show the plot
plt.show()