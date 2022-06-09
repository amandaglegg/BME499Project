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

#%%
# Printing R peaks
# print(_, rpeaks1)
# print(_, rpeaks2)

#convert dictionary to list
data1 = list(rpeaks1.values())
data2 = list(rpeaks2.values())
# print(data1)
# print(data2)

#removes the sampling rate from the list
data1.pop()
print(data1)
data2.pop()
print(data2)

# ST Segment length based on sampling rate (# of samples) (avg duration = 120ms from paper)
STlen = real_freq * 0.12
ST_len = int(STlen)
print(ST_len)

#Fake ST segment length
sampling_rate = 250
STlenf = sampling_rate * 0.12
ST_lenf = int(STlenf)
print(ST_lenf)

# number of samples after the R peak to find start of ST segment
STstartsam = real_freq * 0.06
STstart_sam = int(STstartsam)
print(STstart_sam)

#Fake number of samples 
STstartsamf =  sampling_rate* 0.06
STstart_samf = int(STstartsamf)
print(STstart_samf)

# start of ST Segment after R peak (0.06s after r peak -> experimental value for st segment start)
# to get the index for the start of the ST for every waveform
STstart_index = []
for x in data1:
    STstart_index.append(x + STstart_sam)
    print(STstart_index)  
STstart_indext = tuple(STstart_index)
STstart = ecg_signal_real[STstart_indext] #get the value at every index in the ecg_signal_real
# print(STstart)

#Fake start of ST segment
STstart_indexf = []
for x in data2:
    STstart_indexf.append(x + STstart_samf)
    print(STstart_indexf)  
STstart_indextf = tuple(STstart_indexf)
STstartf = ecg_signal_fake[STstart_indextf] #get the value at every index in the ecg_signal_real
# print(STstartf)

# End of Segment = Start + duration 
STend_index = []
for x in STstart_index: 
    STend_index.append(x + ST_len)
Stend_indext = tuple(STend_index)
STend = ecg_signal_real[Stend_indext] #get the value at every index in the ecg_signal real
#print("Stend")
#print(STend)
#print("ecg signal")
#print(ecg_signal_real)
#print("ststart ")
#print(STstart)

# Fake End of Segment 
STend_indexf = []
for x in STstart_indexf: 
    STend_indexf.append(x + ST_lenf)
Stend_indextf = tuple(STend_indexf)
STendf = ecg_signal_fake[Stend_indextf]

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

# ST slope assesment (up/down/flat) use the angles instead and make the ST slope the end and for loop
i = 0
Valuesf = [0,0,0]
while i < len(STstart):
    if (STend[i] - STstart[i]) < 0: #down
        Valuesf[0] +=1
    elif (STend[i] - STstart[i]) > 0: #up
        Valuesf[1] +=1
    else: #flat
        Valuesf[2] +=1
    i+=1
    
# the slope is the highest of the 3
index_maxf= np.argmax(Valuesf)
if index_maxf == 0:
    STslopef = "downsloping"
elif index_maxf == 1:
    STslopef = "upsloping"
else:
    STslopef = "flat"
print(STslopef)
    





# %%
