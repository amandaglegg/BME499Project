import os
import pandas as pd

os.chdir("..") #move up one directory to BME 499
our_path = os.path.abspath(os.curdir)
pre_ecg_txt_path = our_path + '/datasets/resting_ECG.txt'
post_ecg_txt_path = our_path + '/datasets/exercise_ECG.txt'
rest_ecg = pd.read_csv(pre_ecg_txt_path)
exercise_ecg = pd.read_csv(post_ecg_txt_path)
rest_ecg.to_csv('rest_ecg.csv', 
                  index = None)
exercise_ecg.to_csv('exercise_ecg.csv', 
                  index = None)

print (rest_ecg)