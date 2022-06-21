#Plot raw ECG#
# by Chris and Christine#
#%%
#Import libraries#

import numpy as np
from numpy import mean
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
    # ST Segment length based on sampling rate (# of samples) (avg duration = 120ms from paper)
    STlen = freq * 0.12
    ST_len = int(STlen)
    # number of samples after the R peak to find start of ST segment
    STstartsam = freq * 0.06
    STstart_sam = int(STstartsam)

    # start of ST Segment after R peak (0.06s after r peak -> experimental value for st segment start)
    # to get the index for the start of the ST for every waveform
    STstart_index = []
    for x in data:
        STstart_index.append(x + STstart_sam)
    STstart_indext = tuple(STstart_index)
    STstart = ecg[STstart_indext] #get the value at every index in the ecg_signal_real
    # print(STstart)

    # End of Segment = Start + duration 
    STend_index = []
    for x in STstart_index: 
        STend_index.append(x + ST_len)
    Stend_indext = tuple(STend_index)
    STend = ecg[Stend_indext] #get the value at every index in the ecg_signal real
  
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
stfakeslope = STslope(ecg2, 250)
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

# %% Old peak -> difference between the rest and peak exercise ST depression 

# takes in the baseline for ecg 1 and ecg 2 and the ststart for ecg 1 and ecg2
def old_peak (rest_baseline, exercise_baseline, rest_STstart, exercise_STstart): 
    All_rest = []
    for x in rest_STstart:
        All_rest.append(abs(rest_baseline)- abs(rest_STstart)) #calculates ST depression
    ST_dep_rest = mean(All_rest) # Averages the ST depression for the rest ecg
    All_exercise = []
    for x in exercise_STstart:
        All_exercise.append(abs(exercise_baseline)- abs(exercise_STstart)) #calculates ST depression
    ST_dep_exercise = mean(All_exercise) # Averages the ST depression for the exercise ecg
    OP = (abs(ST_dep_rest)-abs(ST_dep_exercise))
    return OP

#%%
def baseline(ecg, freq): # takes in the ecg signal to recalculate r peaks and calculate p peaks
    # Extract R-peaks locations
    _, rpeaks = nk.ecg_peaks(ecg, sampling_rate = freq)
    # Delineate
    signal, waves = nk.ecg_delineate(ecg, rpeaks, sampling_rate = freq, method="dwt", show=False, show_type='all')
    #convert r peaks dictionary to list
    indexR = list(rpeaks.values())
    #removes the sampling rate from the list
    indexR.pop()
    #Get the values of the r peaks from the ecg 
    dataR = []
    i = 0 
    while i < len(indexR):
        dataR.append(ecg[indexR[i]]) #get the r peak values added to the list
        i += 1
    dataP = waves["ECG_P_Peaks"] # list of the samples that the p peaks occur
    
    ppeak =[] 
    i = 0
    while i < len(dataP):
        ppeak.append (ecg[dataP[i]]) # adds the p peak values to the list
        i += 1
    averages = []
    i = 0 
    while i < len(dataR):
        a = ((dataR[i] + ppeak[i])/2)
        averages.append(a) # a list of the average of ppeaks and rpeaks
        i += 1
    base_volt = mean(averages) # the average of the average values  
    return base_volt

# %% Test for baseline value
a = baseline(ecg1, real_freq)
print(a)
b = baseline(ecg2, 250)
print(b)


# %%
