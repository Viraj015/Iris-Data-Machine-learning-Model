#!/usr/bin/env python
# coding: utf-8

# # IMPORTING LIBRARIES

# In[2]:


#import library 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier,AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_predict,cross_val_score,train_test_split
from sklearn.metrics import classification_report,confusion_matrix,roc_curve,auc,precision_recall_curve,roc_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report
from sklearn.ensemble import GradientBoostingClassifier
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from imblearn.over_sampling import SMOTE
import imblearn
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import TomekLinks
from imblearn.under_sampling import ClusterCentroids
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.simplefilter(action='ignore')


# # IMPORTING DATA

# In[8]:


dataset = pd.read_csv('CH_2016_17.csv')


# **CONSIDERING ONLY THE SUBSET OF THE GIVEN DATASET i.e DATA COLLECTED IN 2016 & 2017.**

# In[9]:


dataset['CreatedDate'] = pd.to_datetime(dataset['CreatedDate'])
dataset = dataset.set_index(dataset['CreatedDate'])
dataset = dataset.sort_index()


# In[10]:


data = dataset['2016-01-01':'2018-01-01']


# In[11]:


data = data.drop('CreatedDate', axis = 1)


# In[12]:


data = data.reset_index('CreatedDate')
data.head(10)
#data = pd.read_csv('Datasets/CH_2016_17.csv')


# # DATA CLEANING + EDA

# In[13]:


#check the lable info
print(data['HighRisk'].value_counts())
print(data['EmergencyReferral'].value_counts())


# In[14]:


#check how many rows * columns
print(data.shape)


# In[15]:


#remove all the missing value in Highrisk

data[['EmergencyReferral','HighRisk']].isnull().sum()


# In[16]:


data[['EmergencyReferral','HighRisk']]


# **NUMBER OF UNIQUE PATIENT ID**

# In[17]:


print("The number of unique patient is ")
len(data["PatientID"].unique())


# In[129]:


df_temp1=data
import time
start = time.time()
dicts = {}
num_keys = range(len(df_temp1["PatientID"].unique()))
values =list(df_temp1["PatientID"].unique())
for i in num_keys:
    #print(values[i])
    dicts[i] = values[i]
#print(dicts)
end = time. time()
print(end - start)


# In[23]:


# Check the different for PID
#data.iloc[:5,:23]

data.iloc[:5, 2:23]


# In[24]:


# Check the different for PID
data.iloc[:5,23:43]


# In[25]:


inv_map = {v: k for k, v in dicts.items()}
df_temp1["PatientID"]=df_temp1["PatientID"].map(inv_map)


# In[26]:


df_temp1.head()


# **CHECK THE CHECKUP DISTRIBUTION**

# In[27]:


ct=pd.DataFrame(df_temp1["PatientID"].value_counts())
ct.reset_index(level=0, inplace=True)
ct.columns=["PatientID","Count"]


# In[28]:


num_checkup=pd.DataFrame(ct["Count"].value_counts())
num_checkup


# **OUTPUT THE CHECKUP DISTRIBUTION**

# In[29]:


num_checkup.to_csv('num_checkup_distribution.csv')


# **OUTPUT THE DATASET WITH FAKE PATIENT ID**

# In[30]:


df_temp1.to_csv('deidentified_pid.csv')


# In[31]:


data = df_temp1.dropna(axis=0, subset=['EmergencyReferral','HighRisk'])


# In[32]:


#Deidentified.csv is the file which stored the data without patient personal information 
#Remove the missing value in HighRisk column because I am going to build the classification model with Highrisk or EmergencyReferral as lable

data.to_csv('deidentified.csv')


# **MAP THE ORIGINAL DATA INTO NUMERICAL DATA**
# 
# The mapping strategy is that if a feature means high risk, then the feature get high score.

# In[33]:


data.columns


# **REMOVING USELESS COLUMNS FOR CLASSIFICATION**

# In[34]:


data=data.drop(columns=['CompanyName','CreatedDate','ZipCode','PatientID','AgeGroup','State','PrimaryCareProvider',
                  'DiabetesPreventionProgram', 'COPDProgram',
       'CoronaryArteryDiseaseProgram', 'DiabetesProgram',
       'HeartFailureProgram', 'AsthmaProgram'])


# In[35]:


data.columns


# **FASTING**

# In[36]:


# Because if the patient did not fast before the test, the result is not reliable.
# Strategy: Remove all rows with the not fasting value instead of mapping it into 1 or 0
#data["Fasting"]=data["Fasting"].map({'YES':1, 'NO':0})
data = data.drop(data[data["Fasting"]=='NO'].index)


# In[37]:


# remove the missing value in Fasting.
data["Fasting"].value_counts()
data = data.drop(data[data["Fasting"].isnull()].index)


# In[38]:


print(data["Fasting"].isnull().sum())


# **SMOKING**

# In[39]:


# There is no No option for 'Smoking','SmokelessTobacco'. Those records are anormal.
data = data.drop(data[data["Smoking"]=='NO'].index)
data = data.drop(data[data['SmokelessTobacco']=='NO'].index)
maps={'NEVER':0, 'QUIT':1, 'YES':2}
data["Smoking"]=data["Smoking"].map(maps)
data['SmokelessTobacco']=data['SmokelessTobacco'].map(maps)


# In[40]:


print(data[['SmokelessTobacco','Smoking']].isnull().sum())


# **ACTIVITY**

# In[41]:


#There is no No option for 'Activity','Activity'. Those records are anormal. just remove it
#Since Activity is required quesiton for patient. There are only about 500 missing values.

data = data.drop(data[data["Activity"].isnull()].index)
maps={'YES':1, 'NO':0}
data["Activity"]=data["Activity"].map(maps)


# In[42]:


print(data["Activity"].isnull().sum())


# **ALCOHOL**

# In[43]:


# alcohol -There are only about 500 missing value.
data = data.drop(data[data["Alcohol"].isnull()].index)
maps={'YES':1, 'NO':0}
data["Alcohol"]=data["Alcohol"].map(maps)
print(data["Alcohol"].isnull().sum())


# **HYPERTENSION HISTORY**

# In[44]:


#"HypertensionHistory" -There are only about 500 missing values.
data = data.drop(data[data["HypertensionHistory"].isnull()].index)
data["HypertensionHistory"]=data["HypertensionHistory"].map({'YES':1, 'NO':0})
print(data["HypertensionHistory"].isnull().sum())


# **DIABETES HISTORY**

# In[45]:


#"DiabetesHistory"-There are only about 500 missing values.
data = data.drop(data[data["DiabetesHistory"].isnull()].index)
data["DiabetesHistory"]=data["DiabetesHistory"].map({'YES':1, 'NO':0})
print(data["DiabetesHistory"].isnull().sum())


#  **CORONARY ARTERY HISTORY**

# In[46]:


#'CoronaryArteryHistory'-There are only about 500 missing values.
data = data.drop(data[data["CoronaryArteryHistory"].isnull()].index)
data["CoronaryArteryHistory"]=data["CoronaryArteryHistory"].map({'YES':1, 'NO':0})
print(data["CoronaryArteryHistory"].isnull().sum())


# **HIGH CHOLESTROL HISTORY**

# In[47]:


#'HighCholesterolHistory'-There are only about 600 missing values.
data = data.drop(data[data['HighCholesterolHistory'].isnull()].index)
data['HighCholesterolHistory']=data['HighCholesterolHistory'].map({'YES':1, 'NO':0})
print(data['HighCholesterolHistory'].isnull().sum())


# **DAYTIME FATIGUE HISTORY, KIDNEY HISTORY, STROKE HISTORY**

# In[48]:


# Since there are too many missing values in 'DaytimeFatigueHistory'(300K = nan),'KidneyHistory','StrokeHistory'
columns_miss = ['DaytimeFatigueHistory','KidneyHistory','StrokeHistory']
data=data.drop(columns=columns_miss)


# In[49]:


data.isnull().sum()


# **NEWLY ASSESED HYPERTENSION, NEWLY ASSESSED DYSLIPIDEMIA, NEWLY ASSESED DIABETES**

# In[50]:


# Remove useless columns -NewlyAssessedHypertension,NewlyAssessedDyslipidemia,NewlyAssessedDiabetes
columns_newly = ['NewlyAssessedHypertension','NewlyAssessedDyslipidemia','NewlyAssessedDiabetes']
data=data.drop(columns=columns_newly)


# In[51]:


data.isnull().sum()


# **HEART DISEASE HISTORY**

# In[52]:


#'HeartDiseaseHistory'-485=NAN
data = data.drop(data[data['HeartDiseaseHistory']==np.nan].index)
data['HeartDiseaseHistory']=data['HeartDiseaseHistory'].map({'YES':1, 'NO':0})
print(data['HeartDiseaseHistory'].isnull().sum())


# **ASTHMA HISTORY**

# In[53]:


#'SnoringHistory'-58575 missing since patients may not be able to detect sorning by themself. Remove the column
#'AsthmaHistory'-58539 missing since patients are able to detect asthma, heartfailure, and COPDHistory
#'HeartFailureHistory'-50K missing 
#'COPDHistory'-50k missing


#df['D'] = np.where((df.A=='blue') & (df.B=='red') & (df.C=='square'), 'succeed')
maps={'YES':1, 'NO':0}
data['AsthmaHistory']=np.where(data['AsthmaHistory'].isnull(),0,data['AsthmaHistory'].map(maps))
print(data['AsthmaHistory'].isnull().sum())


# **COPD HISTORY**

# In[54]:


data['COPDHistory']=np.where(data['COPDHistory'].isnull(),0,data['COPDHistory'].map(maps))
print(data['COPDHistory'].isnull().sum())


# **SNORING HISTORY**

# In[55]:


data=data.drop(columns='SnoringHistory')


# **HEART FAILURE HISTORY**

# In[56]:


data['HeartFailureHistory']=np.where(data['HeartFailureHistory'].isnull(),0,data['HeartFailureHistory'].map(maps))
print(data['HeartFailureHistory'].isnull().sum())


# In[57]:


data.isnull().sum()


# **HYPERTENSION MEDICATION, CHOLESTROL MEDICATION, DIABETES MEDICATION**

# In[58]:


maps={'YES':1, 'NO':0}
cols_med=['HypertensionMedication','CholesterolMedication','DiabetesMedication']
cols_his=['HypertensionHistory','HighCholesterolHistory','DiabetesHistory']
for i in range(3):
    print("missing value for %s: is %.f" %(cols_med[i],data[cols_med[i]].isnull().sum()))
    #print(data[cols_his[i]].value_counts())
    #print("missing value for %s: is %.f" %(cols_his[i],data[cols_his[i]].isnull().sum()))
    data[cols_med[i]]=np.where(data[cols_med[i]].isnull(),999,data[cols_med[i]].map(maps))
    print(data[cols_med[i]].value_counts())


# **STRATEGY TO DEAL WITH HYPERTENSION MEDICATION CHOLESTROL MEDICATION and DIABETES MEDICATION**
# 
# If the history is No, and the medication is Null, then assign 0.
# 
# If the history is Yes, and the medication is Yes, the disease is under controlled, then assign 1.
# 
# If the history is Yes, and the medication is Null or No, the disease is not under controlled, then assign 2.
# 

# In[59]:


#data[cols_med[i]]=data[cols_med[i]].apply(lambda x: 0 if data[cols_his[i]]==0
cols_med=['HypertensionMedication','CholesterolMedication','DiabetesMedication']
cols_his=['HypertensionHistory','HighCholesterolHistory','DiabetesHistory']
for i in range(3):
    data.loc[(data[cols_his[i]]==1) & (data[cols_med[i]]==1), cols_med[i]]=1
    data.loc[(data[cols_his[i]]==1) & (data[cols_med[i]]==0), cols_med[i]]=2
    data.loc[(data[cols_his[i]]==1) & (data[cols_med[i]]==999), cols_med[i]]=2
    data.loc[(data[cols_his[i]]==0) & (data[cols_med[i]]==999), cols_med[i]]=0
    print(data[cols_med[i]].value_counts())


# In[50]:


data.isnull().sum()


# In[60]:


data.shape


# **COTININE**

# In[61]:


# for "COTININE"  -291604 missing 

#data = data.drop(column=COTININE]=="Invalid"].index)
#data = data.drop(data[data["COTININE"]=="Signed Waiver"].index)

#data["COTININE"]=data["COTININE"].map({'Negative':0, 'Positive':1})
#data["COTININE"].value_counts()
data=data.drop(columns=["COTININE"])


# **TEST RESULT** :
# 
# 'GLU','SBP', 'DBP', 'TCHOL','TC:HDL_RATIO', 'LDL', 'HDL', 'TGS', 'AbdominalCir', 'HEIGHT', 'WEIGHT','BMI', 'ALT', 'AST'
# except for 'A1c'
# 
# Remove all the rows with missing value

# In[62]:


cols=['GLU', 'SBP', 'DBP', 'TCHOL',
       'TC:HDL_RATIO', 'LDL', 'HDL', 'TGS', 'AbdominalCir', 'HEIGHT', 'WEIGHT',
       'BMI', 'ALT', 'AST']
data = data.dropna(subset=cols)


# In[63]:


data.shape


# In[64]:


data.isnull().sum()


#  **ASSESSMENT:**
# 
# For 'BPAssessment','DyslipidemiaStatus', 'HDLAssessment', 'LDLAssessment', 'TGSAssessment',
#        'DMAssessment', 'DiabetesStatusUsingGlucose', 
#        'BMIAssessment', 'AbdominalCircumferenceStatus', 'Smoking',
#        'SmokelessTobacco', 'Activity'
#        
# Except for 'DiabetesStatusUsingA1c'
# 
# Remove the rows with missing value

# In[65]:


cols=['BPAssessment','DyslipidemiaStatus', 'HDLAssessment', 'LDLAssessment', 'TGSAssessment',
       'DMAssessment', 'DiabetesStatusUsingGlucose', 'BMIAssessment', 'AbdominalCircumferenceStatus', 'Smoking',
       'SmokelessTobacco', 'Activity']
data = data.dropna(subset=cols)


# In[66]:


data.shape


# In[67]:


data.isnull().sum()


# **MAP THE ASSESSMENT DATA INTO NUMERICAL VALUE ACCORDING TO THE RISK LEVEL.**

# In[68]:


#"BPAssessment"
data["BPAssessment"]=data["BPAssessment"].map({'CONTROLLED':3, 'NORMAL':0, 'HYPERTENSION STG1':2, 'HYPERTENSION STG2':4, 'ELEVATED':1, 'HYPERTENSIVE CRISIS':5})


# In[69]:


#"HDLAssessment"
data = data.drop(data[data["HDLAssessment"]=='NOT FASTING'].index)
data["HDLAssessment"]=data["HDLAssessment"].map({'VERY GOOD':0, 'ACCEPTABLE':1, 'LOW':2})


# In[70]:


#"LDLAssessment"
data = data.drop(data[data["LDLAssessment"]=='NOT FASTING'].index)
data["LDLAssessment"]=data["LDLAssessment"].map({'OPTIMAL':0, 'GOOD':1, 'BORDERLINE HIGH':2, 'CONTROLLED':3, 'HIGH':4}) 


# In[71]:


#"TGSAssessment"
#data = data.drop(data[data["TGSAssessment"]=='NOT FASTING'].index)
data["TGSAssessment"]=data["TGSAssessment"].map({'NORMAL':0, 'BORDERLINE HIGH':1, 'HIGH':2, 'VERY HIGH':3}) 


# In[72]:


#"DMAssessment","DiabetesStatusUsingGlucose","DiabetesStatusUsingA1c"
maps = {'NORMAL':0, 'PRE-DIABETES':2, 'DIABETES':3, 'MANAGED':1}
data = data.drop(data[data["DiabetesStatusUsingGlucose"]=='NOT FASTING'].index)
data["DMAssessment"]=data["DMAssessment"].map(maps)
data["DiabetesStatusUsingGlucose"]=data["DiabetesStatusUsingGlucose"].map(maps)
data["DiabetesStatusUsingA1c"]=data["DiabetesStatusUsingA1c"].map(maps)


# In[73]:


#BMI
maps={'EXTREME OBESITY':4, 'NORMAL':0, 'OBESITY':3, 'OVERWEIGHT':2, 'UNDERWEIGHT':1}
data["BMIAssessment"]=data["BMIAssessment"].map(maps)


# In[74]:


#"AbdominalCircumferenceStatus"
maps={'HIGH RISK':1, 'NORMAL':0}
data["AbdominalCircumferenceStatus"]=data["AbdominalCircumferenceStatus"].map(maps)


# In[75]:


#"Gender"
maps_G={'F':1,'M':0}
data["Gender"]=data["Gender"].map(maps_G)
data["Gender"].isnull().sum()


# In[76]:


data.head(4)


# **CREATE FLAG ACCORDING TO THE RISK LEVEL**

# In[77]:


"""
Create a label feature based on the EmergencyReferral and highrisk for the purpose of building multi-classes models

strategy: 
if EmergencyReferral = yes, and the high risk = yes, assig Flag_Mix = 3
if EmergencyReferral = yes, and the high risk = no, assign Flag_Mix = 2
if EmergencyReferral = no, and the high risk = yes, assign Flag_Mix = 1
if EmergencyReferral = no, and the high risk = no, assign Flag_Mix = 0

"""
data.loc[(data['EmergencyReferral']=='YES') & (data['HighRisk']=='YES'), 'Flag_Mix']=3
data.loc[(data['EmergencyReferral']=='YES') & (data['HighRisk']=='NO'), 'Flag_Mix']=2
data.loc[(data['EmergencyReferral']=='NO') & (data['HighRisk']=='YES'), 'Flag_Mix']=1
data.loc[(data['EmergencyReferral']=='NO') & (data['HighRisk']=='NO'), 'Flag_Mix']=0
print(data['Flag_Mix'].value_counts())


# In[78]:


"""
Create a label feature based on the EmergencyReferral only for the purpose of building bi-classes models

strategy: 
if EmergencyReferral = yes  assig Flag_E = 1
if EmergencyReferral = no, assign Flag_E = 0
"""
data.loc[(data['EmergencyReferral']=='YES'), 'Flag_E']=1
data.loc[(data['EmergencyReferral']=='NO'), 'Flag_E']=0
print(data['Flag_E'].value_counts())


# In[79]:


"""
Create a label feature based on the EmergencyReferral and highrisk for the purpose of building bi-classes models

Strategy: 
if Emergency Referral = yes, or High Risk = yes, assig Flag = 1
if Emergency Referral = yes, and High Risk = no, assign Flag = 0
if Emergency Referral = no, and High Risk = yes, assign Flag = 0
if Emergency Referral = no, and High Risk = no, assign Flag= 0


"""
data.loc[(data['EmergencyReferral']=='YES') | (data['HighRisk']=='YES'), 'Flag']=1
data.loc[(data['EmergencyReferral']=='NO') & (data['HighRisk']=='NO'), 'Flag']=0
print(data['Flag'].value_counts())
print(data['Flag'].isnull().sum())
#data=data.drop(columns=['Flag_EH'])


# In[80]:


data.isnull().sum()


# **WRITE OUT THE CLEANED DATA INTO TWO DATASETS:** 
# 
# 1. With the Gender info for further descriptive analysis.
# 
# 2. Remove all gender & age-related features.

# In[81]:


# with ggender & age-related features:
data.to_csv('cleaned_G.csv',index=False)


# In[82]:


# without gender-related features:
cols=["Fasting","A1c","DiabetesStatusUsingA1c","AbdominalCircumferenceStatus",
      "Mammography","PapSmear","ClinicalBreastExam","ColorectalExam","DiabetesDuringPregnancyHistory"]
data_cleaned=data.drop(columns=cols)
data_cleaned.isnull().sum()


# In[83]:


data_cleaned.to_csv('cleaned.csv',index=False)


# In[73]:


df = pd.read_csv('cleaned.csv')


# # MORE DATA PREPROCESSING

# **SPLITTING DATA SET INTO FEATURES AND LABELS**

# In[74]:


#split dataset into two sets: 1. feature set 2. class
cols=['EmergencyReferral', 'HighRisk', 'Flag_Mix','Flag_E', 'Flag']
X=df.drop(columns=cols)
#class
y=df["Flag"]
X.head()


# In[75]:


X.shape


# **CHECK CLASS DISTRIBUTION--CLASS IMBALANCE**

# In[76]:


# detect class imbalance
df["Flag"].value_counts()


# **SPLITTING DATASET INTO TRAIN-TEST SPLITS**

# In[77]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)


# **APPLYING UNDERSAMPLING METHOD FOR IMBALANCED CLASS**

# In[78]:


import imblearn
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state=0)
X_resampled, y_resampled = rus.fit_resample(X_train, y_train)


# In[79]:


#check the class distribution
import collections
print(sorted(collections.Counter(y_resampled).items()))


# **APPLYING RANDOM FOREST ALGORITHM TO SELECT THE IMPORTANT FEATURES**

# In[80]:


cols=['SBP',
      'DBP',                                      
 'BPAssessment',                             
  'GLU',                                      
  'DiabetesStatusUsingGlucose',               
  'DMAssessment',                            
  'BMI',                                      
  'Age',                                      
  'LDL',                                     
 'TCHOL',                                     
 'WEIGHT',                                   
 'AbdominalCir',                             
 'TGS',                                       
 'AST',                                       
 'ALT',                                      
 'DiabetesHistory',                           
 'TC:HDL_RATIO',                            
 'HEIGHT',                                   
 'HDL',                                    
 'METS_Risks',                                
 'Activity',                                 
 'BMIAssessment',                             
 'Smoking',                                  
 'HypertensionHistory',                       
 'LDLAssessment',                          
 'DyslipidemiaStatus',                        
 'CholesterolMedication',                     
 'HypertensionMedication',                    
 'HDLAssessment',                             
 'Gender',                                    
 'AsthmaHistory',                          
 'SmokelessTobacco',                          
 'TGSAssessment',                           
 'Alcohol',                                  
 'HeartDiseaseHistory',                    
 'HighCholesterolHistory',                    
 'DiabetesMedication',                        
 'COPDHistory',                               
 'HeartFailureHistory',                       
 'CoronaryArteryHistory']


# In[81]:


cols = np.array(cols)


# # homework start here:
# # Feature selection (35 credits)

# ## Perform any three feature selection methods apart from random forest thought in class or from outside to select top 15 appropriate features to build models.
# 
# ## Note: - You can use the same feature selection methods given in the template or something else of your choice. sklearn.feature_selection contains a lot of feature selection methods, feel free to pick 3 of them and run it in our dataset. Or you can use the feature selection code/examples provides during the class.
# 

# In[12]:


# below is an example.

from sklearn.ensemble import RandomForestClassifier

feat_labels = X.columns[:]

forest = RandomForestClassifier(n_estimators=10,
                                random_state=0,
                                n_jobs=-1)

forest.fit(X_resampled, y_resampled)
importances = forest.feature_importances_

indices = np.argsort(importances)[::-1]

for f in range(X_resampled.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 41, 
                            feat_labels[indices[f]], 
                            importances[indices[f]]))

plt.title('Feature Importances')
plt.bar(range(X_resampled.shape[1]), 
        importances[indices],
        color='lightblue', 
        align='center')

plt.xticks(range(X_resampled.shape[1]), 
           feat_labels[indices], rotation=90)
plt.xlim([-1, X_resampled.shape[1]])
plt.tight_layout()
#plt.savefig('./random_forest.png', dpi=300)
plt.show()


# **Seqential forward feature selection using Logistic Regression**
# This is just an example, you can choose whatever methods.

# In[13]:


from sklearn.feature_selection import SequentialFeatureSelector


# In[14]:


# write down some code here.

lr= LogisticRegression(C=100,max_iter=300)
sfs =SequentialFeatureSelector(lr,n_features_to_select=15)
sfs.fit(X_resampled, y_resampled)


# In[15]:


print(
    "Features selected by forward sequential selection: "
    f"{cols[sfs.get_support()]}"
)


# **Feature importance through coefficients**

# In[16]:


from sklearn.linear_model import RidgeCV


# In[37]:


ridge = RidgeCV(alphas=np.logspace(-6, 6, num=5)).fit(X_resampled, y_resampled)
importance = np.abs(ridge.coef_)
indices_1 =  np.argsort(importance)[::-1]

for f in range(X_resampled.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 41, 
                            feat_labels[indices_1[f]], 
                            importance[indices_1[f]]))

plt.title('Feature Importances')
plt.bar(range(X_resampled.shape[1]), 
        importance[indices_1],
        color='lightblue', 
        align='center')

plt.xticks(range(X_resampled.shape[1]), 
           feat_labels[indices_1], rotation=90)
plt.xlim([-1, X_resampled.shape[1]])
plt.tight_layout()
#plt.savefig('./random_forest.png', dpi=300)
plt.show()


# **Seqential feature selection (Backward & Forward)**

# In[33]:


from sklearn.feature_selection import SequentialFeatureSelector
import time  #'from time import time' doesnt work sometimes

# please write down some code here. 
tic_fwd = time.time()
sfs_forward =SequentialFeatureSelector(LogisticRegression(),n_features_to_select=15,direction='forward')
sfs_forward.fit(X_resampled, y_resampled)
toc_fwd = time.time()

tic_bwd = time.time()
sfs_backward =SequentialFeatureSelector(LogisticRegression(),n_features_to_select=15,direction='backward')
sfs_backward.fit(X_resampled, y_resampled)
toc_bwd = time.time()

print(
    "Features selected by forward sequential selection: "
    f"{cols[sfs_forward.get_support()]}"
)
print(f"Done in {toc_fwd - tic_fwd:.3f}s")
print(
    "Features selected by backward sequential selection: "
    f"{cols[sfs_backward.get_support()]}"
)
print(f"Done in {toc_bwd - tic_bwd:.3f}s")


# **Taking the most important columns based on any one of the above feature selection methods**

# ## Please summarize the difference of selected features among different faeture selection methods. (5 points)
# 
# # You answer is here:
# selected most common features from all the three feature selections methods and rest of the features which are not common among any of the selection methods have been selected based on the highest coeffients. 

# In[82]:


X_resampled=pd.DataFrame(X_resampled,columns=X.columns)
X_test=pd.DataFrame(X_test,columns=X.columns)


# In[83]:


cols=['SBP', 'DBP', 'BPAssessment', 'GLU', 'BMI', 'TGS', 'Age', 'TCHOL', 'AST', 'LDL', 'ALT', 'WEIGHT', 'DMAssessment', 'AbdominalCir', 'METS_Risks', 'DiabetesStatusUsingGlucose']
X_resampled=X_resampled[cols]
X_test=X_test[cols]


# **Feature Scaling**

# In[84]:


from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
X_resampled = mms.fit_transform(X_resampled)
X_test = mms.fit_transform(X_test)


# # MODEL BUILDING 

# **LOGISTIC REGRESSION**

# In[85]:


def lr_fit(X_resampled, y_resampled,X_test,y_test):
    lr = LogisticRegression(random_state = 0)
    clf = lr.fit(X_resampled, y_resampled)
    clf_predicted = clf.predict(X_test)
    ConfusionMatrix_Report(y_test=y_test,y_predicted=clf_predicted)
    y_score = lr.decision_function(X_test)
    ROC_decison_function(y_test, y_score)


# **SUPPORT VECTOR MACHINE**

# In[86]:


def svm_rbf_fit(X_resampled, y_resampled,X_test,y_test):
    svm_rbf=SVC(gamma='auto')
    clf = svm_rbf.fit(X_resampled, y_resampled)
    clf_predicted = clf.predict(X_test)
    ConfusionMatrix_Report(y_test=y_test,y_predicted=clf_predicted)
    y_score = svm_rbf.decision_function(X_test)
    ROC_decison_function(y_test, y_score)


# **GRADIENT BOOSTING TREE**

# In[93]:


from sklearn.ensemble import GradientBoostingClassifier
def Gradient_Boosting_fit(X_resampled, y_resampled,X_test,y_test):
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0).fit(X_resampled, y_resampled)
    clf_predicted = clf.predict(X_test)
    y_score = clf.predict_proba(X_test)[:, 1]
    ConfusionMatrix_Report(y_test=y_test,y_predicted=clf_predicted)
    ROC_decison_function(y_test, y_score)


# # MODEL EVALUATION METRICS

# **CONFUSION MATRIX**

# In[88]:


def ConfusionMatrix_Report(y_test,y_predicted):  
    confusion = confusion_matrix(y_test, y_predicted)
    ACC=accuracy_score(y_test, y_predicted)
    Precision=precision_score(y_test, y_predicted)
    Recall=recall_score(y_test, y_predicted)
    F1=f1_score(y_test, y_predicted)
    print('---Confusion Matrix---\n', confusion)
    print('\n   Accuracy: {:.2f}'.format(ACC))
    print('\n   Precision: {:.2f}'.format(Precision))
    print('\n   Recall: {:.2f}'.format(Recall))
    print('\n   F1: {:.2f}'.format(F1))
    print('---Classification Report---')
    print('\n   \n', 
    classification_report(y_test, y_predicted, target_names = ['not 1', '1']))
    print("Metric ")
    print('% 0.2f' % Recall,'% 0.2f' % F1,'% 0.2f' % Precision,'% 0.2f' % ACC)


# **RECIEVER OPERATING CHARACTERSTIC CURVE**

# In[89]:


def ROC_decison_function(y_test, y_score):
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    print('% 0.2f' %roc_auc)
    plt.figure()
    ax = plt.axes()
    ax.set_xlim([-0.01, 1.00])
    ax.set_ylim([-0.01, 1.01])
    ax.plot(fpr, tpr, lw=3, label='(AUC = {:0.2f})'.format(roc_auc))
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.title('classifier ROC curve )', fontsize=16)
    plt.legend(loc='best', fontsize=13)
    plt.plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')
    ax.set_aspect('equal')
    plt.show()


# In[71]:


def ROC_No_decision_function(y_test, clf_predicted):
    false_positive_rate,true_positive_rate,thresholds=roc_curve(y_test,clf_predicted)
    roc_auc=auc(false_positive_rate, true_positive_rate)
    print('% 0.2f' %roc_auc)
    print("-----------")
    plt.figure()
    ax = plt.axes()
    ax.set_xlim([-0.01, 1.00])
    ax.set_ylim([-0.01, 1.01])
    ax.plot(false_positive_rate, true_positive_rate,lw=3, label='(AUC = {:0.2f})'.format(roc_auc))
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.title('classifier ROC curve )', fontsize=16)
    plt.legend(loc='best', fontsize=13)
    plt.plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')
    ax.set_aspect('equal')
    plt.show()


# # MODEL TRAINING / TESTING - BEFORE TUNING

# In[94]:


print("--------------------------------------------Logistic Regression---------------------------------------")
lr_fit(X_resampled, y_resampled,X_test,y_test)
print("--------------------------------------------Support Vector Machine-------------------------------------")
svm_rbf_fit(X_resampled, y_resampled,X_test,y_test)
print("-------------------------------------------Gradient Boosting Tree--------------------------------------------")
Gradient_Boosting_fit(X_resampled, y_resampled,X_test,y_test)


# # MODEL HYPERPARAMETER TUNING

# **TUNING SUPPORT VECTOR MACHINE**

# In[96]:


clf = SVC()
grid_values = {'gamma': [   0.1,1,10], 'kernel' : ['linear', 'rbf'],
          'C':[0.1, 1, 10]}

# default metric to optimize over grid parameters: accuracy
grid_clf_acc = GridSearchCV(clf, param_grid = grid_values,cv=3)
grid_clf_acc.fit(X_resampled, y_resampled)
y_decision_fn_scores_acc = grid_clf_acc.decision_function(X_test) 

print('Grid best parameter (max. accuracy): ', grid_clf_acc.best_params_)
print('Grid best score (accuracy): ', grid_clf_acc.best_score_)

# alternative metric to optimize over grid parameters: AUC
# grid_clf_auc = GridSearchCV(clf, param_grid = grid_values, cv=3,scoring = '?')
# write down some code here, hint is above, just fill in the scoring metric.
grid_clf_auc = GridSearchCV(clf, param_grid = grid_values,cv=3,scoring='roc_auc')
grid_clf_auc.fit(X_resampled, y_resampled)
y_decision_fn_scores_auc = grid_clf_auc.decision_function(X_test) 

print('testing set AUC: ', roc_auc_score(y_test, y_decision_fn_scores_auc))
print('Grid best parameter (max. AUC): ', grid_clf_auc.best_params_)
print('Grid best score (AUC): ', grid_clf_auc.best_score_)

# alternative metric to optimize over grid parameters: F-1 score
# write down some code here
grid_clf_f1 = GridSearchCV(clf, param_grid = grid_values,cv=3,scoring='f1')
grid_clf_f1.fit(X_resampled, y_resampled)
y_decision_fn_scores_f1 = grid_clf_f1.decision_function(X_test) 


print('Grid best parameter (max. F1): ', grid_clf_f1.best_params_)
print('Grid best score F1: ', grid_clf_f1.best_score_)


# **TUNING LOGISTIC REGRESSION**

# In[97]:


from sklearn.linear_model import LogisticRegression
    
clf= LogisticRegression()
grid_values = {'max_iter': [100, 300, 500],'C': [ 0.1, 1, 10], 'penalty': ['l1', 'l2']}

# default metric to optimize over grid parameters: accuracy
grid_clf_acc = GridSearchCV(clf, param_grid = grid_values,cv=3)

grid_clf_acc.fit(X_resampled, y_resampled)
y_decision_fn_scores_acc = grid_clf_acc.decision_function(X_test) 

print('Grid best parameter (max. accuracy): ', grid_clf_acc.best_params_)
print('Grid best score (accuracy): ', grid_clf_acc.best_score_)

# alternative metric to optimize over grid parameters: AUC
grid_clf_auc = GridSearchCV(clf, param_grid = grid_values,cv=3,scoring='roc_auc')
grid_clf_auc.fit(X_resampled, y_resampled)
y_decision_fn_scores_auc = grid_clf_auc.decision_function(X_test) 

print('Testing set AUC: ', roc_auc_score(y_test, y_decision_fn_scores_auc))
print('Grid best parameter (max. AUC): ', grid_clf_auc.best_params_)
print('Grid best score (AUC): ', grid_clf_auc.best_score_)

# alternative metric to optimize over grid parameters: F-1 socore
grid_clf_f1 = GridSearchCV(clf, param_grid = grid_values,cv=3,scoring='f1')
grid_clf_f1.fit(X_resampled, y_resampled)
y_decision_fn_scores_f1 = grid_clf_f1.decision_function(X_test) 


print('Grid best parameter (max. F1): ', grid_clf_f1.best_params_)
print('Grid best score F1: ', grid_clf_f1.best_score_)


# **TUNING GRADIENT BOOSTING TREE**

# In[98]:


#'min_samples_leaf':range(100,1000,10),'max_features':range(2,16,2),\'min_samples_split':range(1000,2000,10),\learning rate:
clf =  GradientBoostingClassifier()
grid_values={'n_estimators':range(10,100,30),'max_depth':range(3,10,3),        'learning_rate':[0.01,0.1,0.5]}
# default metric to optimize over grid parameters: accuracy
grid_clf_acc = GridSearchCV(clf, param_grid = grid_values,cv=3)
grid_clf_acc.fit(X_resampled, y_resampled)
y_decision_fn_scores_auc = grid_clf_auc.predict_proba(X_test)[:, 1] 
print('Grid best parameter (max. accuracy): ', grid_clf_acc.best_params_)
print('Grid best score (accuracy): ', grid_clf_acc.best_score_)
# alternative metric to optimize over grid parameters: AUC
grid_clf_auc = GridSearchCV(clf, param_grid = grid_values,cv=3,scoring='roc_auc')
grid_clf_auc.fit(X_resampled, y_resampled)
y_decision_fn_scores_auc = grid_clf_auc.predict_proba(X_test)[:, 1] 


print('Test set AUC: ', roc_auc_score(y_test, y_decision_fn_scores_auc))
print('Grid best parameter (max. AUC): ', grid_clf_auc.best_params_)
print('Grid best score (AUC): ', grid_clf_auc.best_score_)


# # MODEL TRAINING / TESTING - AFTER HYPERPARAMETER TUNING

# **FITTING LOGISTIC REGRESSION**

# In[99]:


def lr_fit(X_resampled, y_resampled,X_test,y_test):
    lr = LogisticRegression(C=1000, max_iter= 600,penalty= 'l2')
    clf = lr.fit(X_resampled, y_resampled)
    clf_predicted = clf.predict(X_test)
    ConfusionMatrix_Report(y_test=y_test,y_predicted=clf_predicted)
    y_score_lr = lr.decision_function(X_test)
    ROC_decison_function(y_test, y_score=y_score_lr)
    X_resampled=pd.DataFrame(X_resampled,columns=cols)
    print("no inverse scaled coefficient for lr:",clf.coef_)
    feature_importance = abs(clf.coef_[0])
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    featfig = plt.figure()
    featax = featfig.add_subplot(1, 1, 1)
    featax.barh(pos, feature_importance[sorted_idx], align='center')
    featax.set_yticks(pos)
    featax.set_yticklabels(np.array(X_resampled.columns)[sorted_idx], fontsize=8)
    featax.set_xlabel('Relative Feature Importance')
    plt.tight_layout()   
    plt.show()


# In[100]:


lr_fit(X_resampled, y_resampled,X_test,y_test)


# **FITTING GRADIENT BOOSTING TREE**

# In[101]:


from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier(learning_rate=0.2, max_depth=3, n_estimators=60).fit(X_resampled, y_resampled)
clf_predicted = clf.predict(X_test)
y_score = clf.predict_proba(X_test)[:, 1]
ConfusionMatrix_Report(y_test=y_test,y_predicted=clf_predicted)
ROC_decison_function(y_test, y_score)


# In[102]:


def feature_importances_(self):
    total_sum = np.zeros((self.n_features, ), dtype=np.float64)
    for tree in self.estimators_:
        total_sum += tree.feature_importances_ 
    importances = total_sum / len(self.estimators_)
    return importances


# In[103]:


clf = GradientBoostingClassifier(learning_rate=0.2, max_depth=3, n_estimators=60).fit(X_resampled, y_resampled)
clf_predicted = clf.predict(X_test)
y_score = clf.predict_proba(X_test)[:, 1]
importances = clf.feature_importances_

indices = np.argsort(importances)[::-1]
X_resampled= pd.DataFrame(X_resampled)
X_resampled.columns=cols
feat_labels=X_resampled.columns
for f in range(X_test.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 17, 
                            feat_labels[indices[f]], 
                            importances[indices[f]]))
plt.figure(figsize=(8,8), facecolor='white')
plt.title('Feature Importances')
plt.bar(range(X_resampled.shape[1]), 
        importances[indices],
        color='dodgerblue', 
        align='center')

plt.xticks(range(X_resampled.shape[1]), 
           feat_labels[indices], rotation=90)
plt.xlim([-1, X_resampled.shape[1]])
plt.tight_layout()
plt.show()
plt.savefig("Feature_Important_GB")


# **FITTING SUPPORT VECTOR MACHINE**

# In[104]:


def svm_rbf_fit(X_resampled, y_resampled,X_test,y_test):
    svm_rbf = SVC(kernel='rbf', C=10,gamma=1,probability=True)
    clf = svm_rbf.fit(X_resampled, y_resampled)
    clf_predicted = clf.predict(X_test)
    ConfusionMatrix_Report(y_test=y_test,y_predicted=clf_predicted)
    y_score = svm_rbf.decision_function(X_test)
    ROC_decison_function(y_test, y_score)


# In[245]:


svm_rbf_fit(X_resampled, y_resampled,X_test,y_test)


# # Clustering

# In[3]:


dataset = pd.read_csv('Customers.csv')
X = dataset.iloc[:, [3, 4]].values
print(X)


# In[4]:


from sklearn.cluster import KMeans
wcss = []
for i in range(2, 11):
    # write down some code here
    kmeans =  KMeans(n_clusters=i,random_state=1)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)


plt.plot(range(2, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# # There is a method using elbow to select the best hyperparameter on K.
# # https://www.scikit-yb.org/en/latest/api/cluster/elbow.html
# # please use the metric='silhouette'

# ## it requires yellowbrick, you can try  pip install yellowbrick, make sure you use the latest version.

# In[6]:


from yellowbrick.cluster import KElbowVisualizer

# Instantiate a scikit-learn K-Means model
model = KMeans(random_state=0)

# Instantiate the KElbowVisualizer with the number of clusters and the metric 
visualizer = KElbowVisualizer(model, k=(2,11))

# Fit the data and visualize
visualizer.fit(X)    
visualizer.poof() 


# In[7]:


kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)


# 

# In[8]:


from sklearn.metrics import silhouette_score

print(silhouette_score(X, kmeans.labels_))


# In[9]:


plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()


# ## please write down your suggestions on this course: 5 points
# 
# * answer: This course has been designed very well. It would be made better by introducting few more datasets and see if the same algorithm or approach would be the best approach or we might need to explore alternative approach.

# 

# 

# 

# 

# 
