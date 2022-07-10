
#ECG processing -> combining result with additional user data -> run combined csv through model
#To be added: Code for combining ST slope, old peak to user data collected through survey

#%%
#ECG Processing and features
import pathlib
import numpy as np
from numpy import mean
from numpy import nan
import pandas as pd
import matplotlib.pyplot as plt
import os
import neurokit2 as nk
import seaborn as sns
import pickle
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
#%%

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
        STslope = 3 #"downsloping"
    elif index_max == 1:
        STslope = 1 #"upsloping"
    else:
        STslope = 2 #"flat"
    # print("In STslope(), ST slope is:", STslope)
    return STslope

def PR_baseline(ecg, freq):
    '''
    This function takes ecg and sampling frequency and returns the baseline value of PR baselines, 
    this can be used to calculate ST depression
    baseline  = average of ppeak and rpeak
    '''
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
    dataP1 = waves['ECG_P_Peaks'] # list of the samples that the p peaks occur
    #get rid of any invalid entries
    dataP = [x for x in dataP1 if np.isnan(x) == False]
    
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
    baseline = mean(averages) # the average of the average values  
    print("in baseline()")
    return baseline

def old_peak (rest_STstart, exercise_STstart, rest_baseline, exercise_baseline):
    All_rest = []
    i = 0
    while i < len(rest_STstart):
        if rest_baseline < rest_STstart[i]: # tests for if there is elevation
            elevation = 1
        else:
            All_rest.append(abs(rest_baseline)- abs(rest_STstart))
        i += 1    
    ST_dep_rest = mean(All_rest) # Averages the ST depression for the rest ecg
    if ST_dep_rest < 0.1 or elevation == 1:
        nd = 1 #There is no depression, nd = no depression
    else:
        nd = 0 # There is ST depression
    print("Average ST depression for rest ecg", ST_dep_rest)
    elevation = 0
    All_exercise = []
    i = 0
    while i < len(exercise_STstart):
        if exercise_baseline < exercise_STstart[i]:
            elevation = 1
        else: 
            All_exercise.append(abs(exercise_baseline)- abs(exercise_STstart))
        i += 1
    ST_dep_exercise = mean(All_exercise) # Averages the ST depression for the exercise ecg
    if ST_dep_exercise < 0.1 or elevation == 1:
        nd1 = 1 #There is no depression 
    else:
        nd1 = 0 # There is ST depression
    print("Average ST depression for exercise ecg", ST_dep_exercise)
    OP = abs(abs(ST_dep_rest)-abs(ST_dep_exercise))
    if OP < 0.1 or nd and nd1 == 1: #can change this if it becomes 0 only because of exercise
        OP = 0
    print("in oldpeak()")
    return OP

# --- Importing Dataset ---
# df = pd.read_csv("D:/Documents/4B/BME499/github/BME499Project/datasets/ecg_2020-06-01.csv", header=9, usecols = ['Unit'])
os.chdir("..") #move up one directory to BME 499
our_path = os.path.abspath(os.curdir)
ecg_path = our_path + '/datasets/ecg_2020-06-01.csv'
user_path = our_path + '/datasets/fake_user_data.csv'
model_path = our_path + '/code/heart_disease_ETC.pkl'
preexercise_path = our_path + '/datasets/pre_exercise_ecg.csv'
postexercise_path = our_path + '/datasets/post_exercise_ecg.csv'
print("Reading data from: ",ecg_path)

df = pd.read_csv(ecg_path, header=9, usecols = ['Unit'])

# calculate sampling frequency and period
real_freq = len(df)/30
period = 1/ real_freq
# --- Process dataset: convert uV to mV, divide by 1000 ---
# print("original df",df)
df = df/1000
df = df.iloc[:,0].to_numpy()
# print("converted to mV", df)

# Calculating sampling frequency and processing post exercise dataset
dft = pd.read_csv(postexercise_path, header=9, usecols = ['Unit'])
real_freq2 = len(dft)/30
# print("post exercise df",dft)
dft = dft/1000
dft = dft.iloc[:,0].to_numpy()
# print("converted to mV", dft)

# ecg datas to work with are below
ecg1 = df
ecg2 = dft
# compute start and end of ST for ecg1
STstart1,STend1 = STprocess(ecg1,real_freq)
print("returned ST start1:",STstart1)
print("returned ST end1:",STend1)
# compute start and end of ST for excercise ecg (ECG2)
STstart2,STend2 = STprocess(ecg2,real_freq2)
print("returned ST start2:",STstart2)
print("returned ST end2:",STend2)
#compute baselines of ecg 1 and 2, for OP
baseline1 = PR_baseline(ecg1,real_freq)
# baseline2 = PR_baseline(ecg2, 250)
baseline2 = PR_baseline(ecg2, real_freq2)
print("returned baseline 1", baseline1)
print("returend baseline 2", baseline2)
#compute st slope of ecg1
stslope = STslope(STstart1,STend1)

print("returned slope", stslope)
#compute oldpeak using ecg1 and 2
OP = old_peak(STstart1, STstart2, baseline1, baseline2)
print("returend OP", OP)

#%% Combined data (From user csv and ecg processing) 
'''
This part combines ST slope and old peak into the user data, 
and outputs a list with processed features, all in numerical values
'''
print("Reading data from: ",user_path)
df1 = pd.read_csv(user_path) #this csv is the one with the age,sex etc..
print(df1)
df1.insert(7,'oldpeak', OP, allow_duplicates = False)
df1.insert(8, 'ST_Slope', stslope, allow_duplicates = False)
df1 = df1.dropna(how='all', axis='columns')
print(df1)

data = df1.values.tolist()
data = MinMaxScaler().fit_transform(data)

# %% Running complete data through model 
file = open(model_path,'rb')
model = pickle.load(file)
file.close()

# --- Prediction using ET Classifier Boosting Model ---
result = model.predict(data)

# --- Print Heart Disease Status ---
if result[0] == 1:
  print('\033[1m' + '.:. Heart Disease Detected!.:.' + '\033[0m')
else:
  print('\033[1m' + '.:. Heart Disease Not Detected!.:.' + '\033[0m')

# %%
