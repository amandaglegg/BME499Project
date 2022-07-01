#%%

import pickle

infile = open('heart_disease_ETC.pkl','rb')
model = pickle.load(infile)
infile.close()

# %% test
# --- Turn Information into List ---
data = [[0.714, 1, 0.33, 0.695,   
         0.5, 0.4788, 1, 0.4318,         
         0.5]]           

# --- Prediction using ET Classifier Boosting ---
result = model.predict(data)
# result = ETclassifier.predict(data)

# --- Print Heart Disease Status ---
if result[0] == 1:
  print('\033[1m' + '.:. Heart Disease Detected!.:.' + '\033[0m')
else:
  print('\033[1m' + '.:. Heart Disease Not Detected!.:.' + '\033[0m')
# %%
