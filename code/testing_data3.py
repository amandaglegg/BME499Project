# %%
# --- Importing Libraries ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import yellowbrick
import os
import pickle

from matplotlib.collections import PathCollection
from statsmodels.graphics.gofplots import qqplot
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
from yellowbrick.classifier import PrecisionRecallCurve, ROCAUC, ConfusionMatrix
from yellowbrick.style import set_palette
from yellowbrick.model_selection import LearningCurve, FeatureImportances
from yellowbrick.contrib.wrapper import wrap

# --- Libraries Settings ---
warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.rcParams['figure.dpi']=100
set_palette('dark')

# %% 
# --- Importing Dataset ---
#df = pd.read_csv("D:/Documents/4B/BME499/test/BME499Project/datasets/heart.csv")

os.chdir("..") #move up one directory to BME 499
our_path = os.path.abspath(os.curdir)
user_path = our_path + '/datasets/fake_user_data.csv'
our_path = our_path + '/datasets/hch.csv'
print(our_path)

df = pd.read_csv(our_path)
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_', regex=True)
# --- Reading Dataset ---
df.head().style.background_gradient(cmap='Reds').set_properties(**{'font-family': 'Segoe UI'}).hide_index() 

#%%
# --- Drop Data Columns ---

df = df.drop(columns=['cholesterol', 'fasting_blood_sugar', 'resting_ecg'])

#%%
# remapping variable names to correct those from previous data set
df['st_slope'].value_counts()
#remap_dict = {1: 'Up', 2: 'Flat', 3: 'Down'}
#df['st_slope'] = df['st_slope'].replace(remap_dict)
df = df.loc[df['st_slope'] != 0].copy() 

# %%
# --- Seperating Dependent Features ---
x = df.drop(['target'], axis=1)
y = df['target']

# --- Data Normalization using Min-Max Method ---
# x = MinMaxScaler().fit_transform(x)

# --- Splitting Dataset into 80:20 ---
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)

#%% 
# --- correlation matrix
corr = df.corr()
corr.style.background_gradient(cmap='coolwarm')
    
# %%
# --- 8.1 Applying Logistic Regression ---
LRclassifier = LogisticRegression(max_iter=1000, random_state=1, solver='liblinear', penalty='l1')
LRclassifier.fit(x_train, y_train)

y_pred_LR = LRclassifier.predict(x_test)

# --- LR Accuracy ---
LRAcc = accuracy_score(y_pred_LR, y_test)

# %%
# --- 8.2 Applying KNN ---
KNNClassifier = KNeighborsClassifier(n_neighbors=3)
KNNClassifier.fit(x_train, y_train)

y_pred_KNN = KNNClassifier.predict(x_test)
# --- KNN Accuracy ---
KNNAcc = accuracy_score(y_pred_KNN, y_test)

# %%
# --- 8.3 Applying SVM ---
SVMclassifier = SVC(kernel='linear', max_iter=1000, C=10, probability=True)
SVMclassifier.fit(x_train, y_train)

y_pred_SVM = SVMclassifier.predict(x_test)
# --- SVM Accuracy ---
SVMAcc = accuracy_score(y_pred_SVM, y_test)

# %%
# ---8.4  Applying Gaussian NB ---
GNBclassifier = GaussianNB(var_smoothing=0.1)
GNBclassifier.fit(x_train, y_train)

y_pred_GNB = GNBclassifier.predict(x_test)

# --- GNB Accuracy ---
GNBAcc = accuracy_score(y_pred_GNB, y_test)

# %%
# --- 8.5 Applying Decision Tree ---
DTCclassifier = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, criterion='entropy', min_samples_split=5,
                                       splitter='random', random_state=1)

DTCclassifier.fit(x_train, y_train)
y_pred_DTC = DTCclassifier.predict(x_test)

# --- Decision Tree Accuracy ---
DTCAcc = accuracy_score(y_pred_DTC, y_test)

# %%
# --- 8.6 Applying Random Forest ---
# Hypertuned RF
RFclassifier = RandomForestClassifier(n_estimators=400, max_features=2, min_samples_split=2, bootstrap = True, max_depth = 100, min_samples_leaf = 1)
# Untuned RF
# RFclassifier = RandomForestClassifier(n_estimators=400, max_features=2, min_samples_split=2, bootstrap = True, max_depth = 70, min_samples_leaf = 1)

RFclassifier.fit(x_train, y_train)
y_pred_RF = RFclassifier.predict(x_test)

# --- Random Forest Accuracy ---
RFAcc = accuracy_score(y_pred_RF, y_test)

# --- Performance Evaluation ---
print('\n\033[1m'+'.: Performance Evaluation'+'\033[0m')
print('*' * 26)
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

# --- Random Forest Confusion Matrix ---
rfcmatrix = ConfusionMatrix(RFclassifier, ax=ax1, cmap='PuRd',
                            title='Random Forest Confusion Matrix')
rfcmatrix.fit(x_train, y_train)
rfcmatrix.score(x_test, y_test)
rfcmatrix.finalize()

# --- Random Forest ROC AUC ---
rccrocauc = ROCAUC(RFclassifier, classes=['False', 'True'], ax=ax2,
                   title='Random Forest ROC AUC Plot')
rccrocauc.fit(x_train, y_train)
rccrocauc.score(x_test, y_test)
rccrocauc.finalize()

# --- Random Forest Learning Curve ---
rcclc = LearningCurve(RFclassifier, ax=ax3, title='Random Forest Learning Curve')
rcclc.fit(x_train, y_train)
rcclc.finalize()

# --- Random Forest Precision Recall Curve ---
rcccurve = PrecisionRecallCurve(RFclassifier, ax=ax4, ap_score=True, iso_f1_curves=True, 
                                title='Random Forest Precision-Recall Curve')
rcccurve.fit(x_train, y_train)
rcccurve.score(x_test, y_test)
rcccurve.finalize()

plt.tight_layout();

# %%
# ---8.7 Applying ET ---
# ETclassifier = ExtraTreesClassifier(n_estimators=200, max_depth=70, max_features=11, min_samples_leaf=2, min_samples_split=2)
# Hypertuned
# ETclassifier = ExtraTreesClassifier(bootstrap = True, n_estimators= 400, random_state = 5, max_depth = 40)
ETclassifier = ExtraTreesClassifier(n_estimators=15, random_state=47)
ETclassifier.fit(x_train, y_train)
y_pred_ET = ETclassifier.predict(x_test)

# --- ET Accuracy ---
ETAcc = accuracy_score(y_pred_ET, y_test)

# --- Performance Evaluation ---
print('\n\033[1m'+'.: Performance Evaluation'+'\033[0m')
print('*' * 26)
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

# --- ET Confusion Matrix ---
etcmatrix = ConfusionMatrix(ETclassifier, ax=ax1, cmap='PuRd',
                            title='Extra Tree Classifier Confusion Matrix')
etcmatrix.fit(x_train, y_train)
etcmatrix.score(x_test, y_test)
etcmatrix.finalize()

# --- ET ROC AUC ---
etrocauc = ROCAUC(ETclassifier, classes=['False', 'True'], ax=ax2,
                  title='Extra Tree Classifier ROC AUC Plot')
etrocauc.fit(x_train, y_train)
etrocauc.score(x_test, y_test)
etrocauc.finalize()

# --- ET Learning Curve ---
etlc = LearningCurve(ETclassifier, ax=ax3, title='Extra Tree Classifier Learning Curve')
etlc.fit(x_train, y_train)
etlc.finalize()

# --- ET Precision Recall Curve ---
etpc = PrecisionRecallCurve(ETclassifier, ax=ax4, ap_score=True, iso_f1_curves=True, 
                            title='Extra Tree Classifier Precision-Recall Curve')
etpc.fit(x_train, y_train)
etpc.score(x_test, y_test)
etpc.finalize()

plt.tight_layout();


# %%
# --- 8.8 Applying Gradient Boosting ---
GBclassifier = GradientBoostingClassifier(random_state=1, n_estimators=100, max_leaf_nodes=3, loss='exponential', 
                                          min_samples_leaf=20)

GBclassifier.fit(x_train, y_train)
y_pred_GB = GBclassifier.predict(x_test)

# --- Gradient Boosting Accuracy ---
GBAcc = accuracy_score(y_pred_GB, y_test)

# %%# --- 8.9 Applying AdaBoost ---
ABclassifier = AdaBoostClassifier(n_estimators=3)

ABclassifier.fit(x_train, y_train)
y_pred_AB = ABclassifier.predict(x_test)

# --- AdaBoost Accuracy ---
ABAcc = accuracy_score(y_pred_AB, y_test)

# %%
# --- Create Accuracy Comparison Table ---
compare = pd.DataFrame({'Model': ['Logistic Regression', 'K-Nearest Neighbour', 'Support Vector Machine', 
                                  'Gaussian Naive Bayes', 'Decision Tree', 'Random Forest', 'Gradient Boosting', 
                                  'AdaBoost','Extra Tree Classifier'], 
                        'Accuracy': [LRAcc*100, KNNAcc*100, SVMAcc*100, GNBAcc*100, DTCAcc*100, RFAcc*100, GBAcc*100, 
                                     ABAcc*100, ETAcc*100]})

# --- Create Accuracy Comparison Table ---
compare.sort_values(by='Accuracy', ascending=False).style.background_gradient(cmap='PuRd').hide_index().set_properties(**{'font-family': 'Segoe UI'})

#%% Create new model
# --- Transform Test Set & Prediction into New Data Frame ---
test = pd.DataFrame(x_test, columns=['age', 'sex', 'chest_pain_type', 'resting_bp_s', 'max_heart_rate', 'exercise_angina', 'oldpeak', 'st_slope'])
pred = pd.DataFrame(y_pred_GB, columns=['target'])
prediction = pd.concat([test, pred], axis=1, join='inner')
# prediction = prediction.drop(['cholesterol', 'fasting_blood_sugar'], axis = 1)

#%% # --- Display Prediction Result ---
prediction.head().style.background_gradient(cmap='Reds').hide_index().set_properties(**{'font-family': 'Segoe UI'})
# --- Export Prediction Result into csv File ---
prediction.to_csv('prediction_heart_disease.csv', index=False)

#%% # --- Export not tuned ET model to Pickle File ---
file = open('ETC_model.pkl', 'wb')
pickle.dump(ETclassifier, file)
file.close()

#%% Testing
infile = open('ETC_model.pkl','rb')
model = pickle.load(infile)
infile.close()

df1 = pd.read_csv(user_path) 
df1.insert(6,'oldpeak', 0, allow_duplicates = False)
df1.insert(7, 'st_slope', 1, allow_duplicates = False)
df1 = df1.dropna(how='all', axis='columns')
data = df1.values.tolist()
print("this is the data going through the model:", data)

# --- Prediction using ET Classifier Boosting Model ---
result = model.predict(data)

# --- Print Heart Disease Status ---

if result[0] == 1:
   print('\033[1m' + '.:. Heart Disease Detected!.:.' + '\033[0m')
else:
   print('\033[1m' + '.:. Heart Disease Not Detected!.:.' + '\033[0m')

#%% Add the heat map
black_grad = ['#100C07', '#3E3B39', '#6D6A6A', '#9B9A9C', '#CAC9CD']
# pd.DataFrame(df, index = ["Age","Sex","Chest Pain Type", "Resting Systolic Blood Pressure","Max Heart Rate","Exercise Angina", "Old Peak","ST Slope","Target"], columns= ["Age","Sex","Chest Pain Type", "Resting Systolic Blood Pressure","Max Heart Rate","Exercise Angina", "Old Peak","ST Slope","Target"])
plt.figure(figsize=(14, 9))
sns.set(font_scale=1.4)
sns.heatmap(df.corr(), annot=True, cmap='OrRd', linewidths=0.1)
plt.suptitle('Correlation Map of Numerical Variables', fontweight='heavy', 
             x=0.03, y=0.98, ha='left', fontsize='16', fontfamily='sans-serif', 
             color=black_grad[0])
# plt.title('Resting blood pressure and "oldpeak" have moderate relationship with age.', 
          # fontsize='10', fontfamily='sans-serif', loc='left', color=black_grad[1])
plt.tight_layout(rect=[0, 0.04, 1, 1.01])
# %%
