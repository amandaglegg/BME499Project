#Plot raw ECG#
# by Chris and Christine#

#Import libraries#

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import neurokit2 as nk
import seaborn as sns
from sklearn import preprocessing

plt.rcParams['figure.figsize'] = [8, 5]  # Bigger images

def STslope(ecg,freq):
    '''
    This function takes ecg dataset and sampling freq to calculate ST slope, taken from christine's code
    '''
    # Extract R-peaks locations
    _, rpeaks = nk.ecg_peaks(ecg, sampling_rate = freq)
    # Delineate
    signal, waves = nk.ecg_delineate(ecg, rpeaks, sampling_rate = freq, method="dwt", show=True, show_type='all')
    #convert dictionary to list
    data = list(rpeaks.values())
    #removes the sampling rate from the list
    data.pop()
    print(data)
    # ST Segment length based on sampling rate (# of samples) (avg duration = 120ms from paper)
    STlen = freq * 0.12
    ST_len = int(STlen)
    print(ST_len)
    # number of samples after the R peak to find start of ST segment
    STstartsam = freq * 0.06
    STstart_sam = int(STstartsam)
    print(STstart_sam)

    # start of ST Segment after R peak (0.06s after r peak -> experimental value for st segment start)
    # to get the index for the start of the ST for every waveform
    STstart_index = []
    for x in data:
        STstart_index.append(x + STstart_sam)
        print(STstart_index)  
    STstart_indext = tuple(STstart_index)
    STstart = ecg[STstart_indext] #get the value at every index in the ecg_signal_real
    # print(STstart)

    # End of Segment = Start + duration 
    STend_index = []
    for x in STstart_index: 
        STend_index.append(x + ST_len)
    Stend_indext = tuple(STend_index)
    STend = ecg[Stend_indext] #get the value at every index in the ecg_signal real
    #print("Stend")
    #print(STend)
    #print("ecg signal")gi
    #print(ecg_signal_real)
    #print("ststart ")
    #print(STstart)

    # ST slope assesment (up/down/flat) use the angles instead and make the ST slope the end and for loop
    i = 0
    Values = [0,0,0]
    while i < len(STstart):
        if (STend[i] - STstart[i]) < 0: #down
            Values[0] +=1
        elif (STend[i] - STstart[i]) > 0: #up
            Values[1] +=1
        else: #flat
            Values[2] +=1
        i+=1
        
    # the slope is the highest of the 3
    index_max= np.argmax(Values)
    if index_max == 0:
        STslope = "downsloping"
    elif index_max == 1:
        STslope = "upsloping"
    else:
        STslope = "flat"
    print(STslope)
    
    return STslope

# --- Importing Dataset ---
# df = pd.read_csv("D:/Documents/4B/BME499/github/BME499Project/datasets/ecg_2020-06-01.csv", header=9, usecols = ['Unit'])
os.chdir("..") #move up one directory to BME 499
our_path = os.path.abspath(os.curdir)
our_path = our_path + '/datasets/ecg_2020-06-01.csv'
print(our_path)
df = pd.read_csv(our_path, header=9, usecols = ['Unit'])
# calculate sampling frequency and period
real_freq = len(df)/30
period = 1/ real_freq
# --- Process dataset: convert uV to mV, divide by 1000 ---
print("original df",df)
df = df/1000
df = df.iloc[:,0].to_numpy()
print("converted to mV", df)

# Similate fake data for comparison
# Generate 15 seconds of ECG signal (recorded at 250 samples / second)
ecgSIM = nk.ecg_simulate(duration=15, sampling_rate=250, heart_rate=70)

ecg1 = df
ecg2 =ecgSIM

stslope = STslope(ecg1,real_freq)
print("returned slope", stslope)

'''
#plot ECG#
x = np.arange(0, 30, period)
plt.plot(x, ecg1.to_numpy())
# naming the x axis
plt.xlabel('Time(s)')
# naming the y axis
plt.ylabel('milliVolts (mV)')
# giving a title to my graph
plt.title('raw ECG graph')
# function to show the plot
plt.show()
'''


        
