#Plot raw ECG#
# by Chris and Christine#
#%%
#Import libraries#

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import neurokit2 as nk
import seaborn as sns
from sklearn import preprocessing


plt.rcParams['figure.figsize'] = [8, 5]  # Bigger images

def STprocess(ecg, freq):
    '''
    this function takes ecg and sampling frequency and return an array of values of start of ST segments
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
       # print("ST start values", STstart)
     # End of Segment = Start + duration 
    STend_index = []
    for x in STstart_index: 
        STend_index.append(x + ST_len)
    Stend_indext = tuple(STend_index)
    STend = ecg[Stend_indext] #get the value at every index in the ecg_signal real
    print("at the end of ST process(), try printing ST end values", STend)
    return STstart, STend

def STslope(STstart,STend):
    '''
    This function takes values of ecg ST segments (START AND END) 
    to calculate ST slope
    '''
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
    print("In STslope(), ST slope is:", STslope)
    return STslope

def PR_baseline(ecg, freq):
    '''
    This function takes ecg and sampling frequency and returns the baseline value of PR baselines, 
    this can be used to calculate ST depression
    baseline  = average of ppeak and rpeak
    '''
    baseline = 1
    print("in baseline()")
    return baseline

def old_peak (rest_STstart, exercise_STstart, rest_baseline, exercise_baseline):
    # %% Old peak -> difference between the rest and peak exercise ST depression 
    # measured 0.8ms after the J point (in mV)
    # Find the st segment in each one 
    # a = ST depression in rest ecg = |STstart| - |STend|
    # b = ST depression in exercise ecg = |STstart| - |STend|
    # OP = |a| - |b| 
    OP = 1
    '''
    If ST depression is measured by baseline to j point
    1. Find STstart in each ecg 
    2. Baseline is PR segment -> take the average of the Ppeaks and Ronsets?
    3. a = ST depression in rest ecg = |baseline| - |STstart|
    4. b = ST depression in exercise ecg = |baseline| - |STstart|
    5. OP = |a| - |b|
    '''
    print("in oldpeak()")
    return OP
#%%
# --- Importing Dataset ---
# df = pd.read_csv("D:/Documents/4B/BME499/github/BME499Project/datasets/ecg_2020-06-01.csv", header=9, usecols = ['Unit'])
os.chdir("..") #move up one directory to BME 499
our_path = os.path.abspath(os.curdir)
our_path = our_path + '/datasets/ecg_2020-06-01.csv'
print("Reading data from: ",our_path)

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

# ecg datas to work with are below
ecg1 = df
ecg2 = ecgSIM
# compute start and end of ST for ecg1
STstart1,STend1 = STprocess(ecg1,real_freq)
print("returned ST start1:",STstart1)
print("returned ST end1:",STend1)
# compute start and end of ST for excercise ecg (ECG2)
STstart2,STend2 = STprocess(ecg2,250)
print("returned ST start2:",STstart2)
print("returned ST end2:",STend2)
#compute baselines of ecg 1 and 2, for OP
baseline1 = PR_baseline(ecg1,real_freq)
baseline2 = PR_baseline(ecg2, 250)
print("returned baseline 1", baseline1)
print("returend baseline 2", baseline2)
#compute st slope of ecg1
stslope = STslope(STstart1,STend1)
print("returned slope", stslope)
#compute oldpeak using ecg1 and 2
OP = old_peak(STstart1, STstart2, baseline1, baseline2)
print("returend OP", OP)
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


        


