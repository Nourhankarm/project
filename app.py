#!/usr/bin/env python
# coding: utf-8

# # **Import library!**

# In[2]:


# Import Neccessary libraries
import numpy as np
import pandas as pd

# Import Visualization libraries
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

#Import Model
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import  LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier

#import warning libraries
import warnings
warnings.filterwarnings('ignore')


# # **Read Data**

# In[3]:


diabetes_data=pd.read_csv('diabetes_prediction_dataset.csv')
diabetes_data.head(20)


# # **EDA**

# In[4]:


# define the data type for each columns and null
diabetes_data.info()


# In[5]:


# define the data type for each columns
diabetes_data.dtypes


# **we have to convert the catigorical data to numerical data**

# In[6]:


#check if there is dublicated data
diabetes_data.duplicated()


# **we should remove duplicates**

# In[8]:


#check the number of duplicated data
diabetes_data.duplicated().sum()


# In[9]:


#remove duplicates
diabetes_data.drop_duplicates(inplace=True)
diabetes_data.duplicated().sum()


# In[10]:


#check if there is missing values
diabetes_data.isna().sum()


#  **no missing values**

# In[11]:


#checking the number of rows and columns of the dataset
diabetes_data.shape


# In[12]:


#generate descriptive statistics
diabetes_data.describe()


# In[13]:


#count the unique values in diabetes column
diabetes_data['diabetes'].value_counts()


# In[14]:


#count the unique values in smoking_history column
diabetes_data['smoking_history'].value_counts()


# In[15]:


#count the unique values in heart_disease column
diabetes_data['heart_disease'].value_counts()


# In[16]:


#count the unique values in gender column
diabetes_data['gender'].value_counts()


# # **Data Visalustion**

# In[17]:


diabetes_data.hist(figsize=(8,8), color='purple')


# In[18]:


#plt.figure(figsize=(9,7))
#correlation=diabetes_data.corr()
#sns.heatmap(correlation,annot=True ,fmt='.2f' ,cbar=True ,cmap='summer')


# In[19]:


fig, ax = plt.subplots(1, 2, figsize=(9, 3))
# Countplot
sns.countplot(x=diabetes_data['gender'],data=diabetes_data ,palette='magma' ,ax=ax[0])
ax[0].set_title(f'Countplot for gender')
# Pie plot
data_counts = diabetes_data['gender'].value_counts()
ax[1].pie(data_counts, labels=data_counts.index, autopct='%1.4f%%', startangle=90, colors=sns.color_palette('vlag'))
ax[1].set_title(f'Pie plot for gender')
plt.show()


# In[20]:


fig, ax = plt.subplots(1, 2, figsize=(9, 3))
# Countplot
sns.countplot(x=diabetes_data['hypertension'],data=diabetes_data ,palette='magma' ,ax=ax[0])
ax[0].set_title(f'Countplot for hypertension')
# Pie plot
data_counts = diabetes_data['hypertension'].value_counts()
ax[1].pie(data_counts, labels=data_counts.index, autopct='%1.4f%%', startangle=90, colors=sns.color_palette('vlag'))
ax[1].set_title(f'Pie plot for hypertension')
plt.show()


# In[21]:


fig, ax = plt.subplots(1, 2, figsize=(9, 3))
# Countplot
sns.countplot(x=diabetes_data['heart_disease'],data=diabetes_data ,palette='magma' ,ax=ax[0])
ax[0].set_title(f'Countplot for heart_disease')
# Pie plot
data_counts = diabetes_data['heart_disease'].value_counts()
ax[1].pie(data_counts, labels=data_counts.index, autopct='%1.4f%%', startangle=90, colors=sns.color_palette('vlag'))
ax[1].set_title(f'Pie plot for heart_disease')
plt.show()


# In[22]:


fig, ax = plt.subplots(1, 2, figsize=(12, 5))
# Countplot
sns.countplot(x=diabetes_data['smoking_history'],data=diabetes_data ,palette='magma' ,ax=ax[0])
ax[0].set_title(f'Countplot for smoking_history')
# Pie plot
data_counts = diabetes_data['smoking_history'].value_counts()
ax[1].pie(data_counts, labels=data_counts.index, autopct='%1.4f%%', startangle=90, colors=sns.color_palette('vlag'))
ax[1].set_title(f'Pie plot for smoking_history')
plt.show()


# In[23]:


fig, ax = plt.subplots(1, 2, figsize=(9, 3))
# Countplot
sns.countplot(x=diabetes_data['diabetes'],data=diabetes_data ,palette='magma' ,ax=ax[0])
ax[0].set_title(f'Countplot for diabetes')
# Pie plot
data_counts = diabetes_data['diabetes'].value_counts()
ax[1].pie(data_counts, labels=data_counts.index, autopct='%1.4f%%', startangle=90, colors=sns.color_palette('vlag'))
ax[1].set_title(f'Pie plot for diabetes')


# In[24]:


print(diabetes_data['age'].mean())


# **Average range of people with diabetes in age 41**
# 
# 

# In[25]:


blood_glucose_level_above_70_and_less_than_100 = diabetes_data[(diabetes_data['blood_glucose_level'] <= 100) & (diabetes_data['blood_glucose_level'] >= 70)]
blood_glucose_level_above_70_and_less_than_100['blood_glucose_level'].value_counts().plot(kind='bar', color='#F89089')
plt.title('normal fasting blood glucose concentration are between 70 mg/dL (3.9 mmol/L) and 100 mg/dL (5.6 mmol/L)')
plt.xlabel('blood_glucose_level')
plt.ylabel('Count of patient')
plt.show()


# **The expected values for normal fasting blood glucose concentration are between 70 mg/dL (3.9 mmol/L) and 100 mg/dL (5.6 mmol/L). When fasting blood glucose is between 100 to 125 mg/dL (5.6 to 6.9 mmol/L) changes in lifestyle and monitoring glycemia are recommended ** **bold text**

# In[26]:


blood_glucose_level_above_200=diabetes_data[diabetes_data['blood_glucose_level']>=200]
blood_glucose_level_above_200['blood_glucose_level'].value_counts().plot(kind='bar', color='#F89089')
plt.title(' the blood glucose concentration are above 200 mg/dl ,mean that the patient is diabetic')
plt.xlabel('blood_glucose_level')
plt.ylabel('Count of patient')
plt.show()


# **A normal A1C level is below 5.7%, a level of 5.7% to 6.4% indicates prediabetes, and a level of 6.5% or more indicates diabetes.**

# In[27]:


Normal_HbA1c_level=diabetes_data[diabetes_data['HbA1c_level']<=5.7]
Normal_HbA1c_level['blood_glucose_level'].value_counts().plot(kind='bar', color='#11B198')
plt.title(' Normal_HbA1c_level')
plt.xlabel('HbA1c_level')
plt.ylabel('Count of patient')
plt.show()


# In[28]:


diabetic_HbA1c_level=diabetes_data[diabetes_data['HbA1c_level']>=6.5]
diabetic_HbA1c_level['blood_glucose_level'].value_counts().plot(kind='bar', color='#11B198')
plt.title('diabetic_HbA1c_level')
plt.xlabel('HbA1c_level')
plt.ylabel('Count of patient')
plt.show()


# **Is hypertension more common in males or females?
# A greater percentage of men (50%) have high blood pressure than women (44%)**
# 

# ***note that the number of females in this data set is 56161 ,and male is 39967 ***

# In[29]:


plt.figure(figsize=(6, 3))
sns.countplot(x=diabetes_data['gender'], hue=diabetes_data['hypertension'], data=diabetes_data ,palette='vlag' )
plt.title('countplot of male and female with respect to hypertension')
plt.show()


# **Worldwide, an estimated 17.7 million more men than women have diabetes mellitus.**

# In[30]:


plt.figure(figsize=(10, 4))
sns.countplot(x=diabetes_data['gender'], hue=diabetes_data['diabetes'], data=diabetes_data ,palette='vlag' )
plt.title('countplot of male and female with respect to diabetes')
plt.show()


# ** regular smokers have a 15-30% higher risk of developing diabetes**

# In[31]:


plt.figure(figsize=(10, 7))
sns.countplot(x=diabetes_data['smoking_history'], hue=diabetes_data['diabetes'], data=diabetes_data ,palette='vlag' )
plt.title('countplot of smoking history with respect to diabetes')
plt.show()


# # **Data preprocessing**

# In[7]:


#converting categorical data into numerical data
encoder=LabelEncoder()
diabetes_data['gender']=encoder.fit_transform(diabetes_data['gender'])
diabetes_data['smoking_history']=encoder.fit_transform(diabetes_data['smoking_history'])


# In[8]:


diabetes_data.head()


# # **Spliting Data**

# In[9]:


X = diabetes_data.drop('diabetes', axis=1)
y = diabetes_data['diabetes']


# In[10]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size = 0.2)


# In[11]:


X_train.shape,X_test.shape,X.shape


# # **LogisticRegression Model**

# In[37]:


lr=LogisticRegression(max_iter=3000)
lr.fit(X_train,y_train)


# In[38]:


y_predection=lr.predict(X_test)


# In[39]:


lr_accuracy = accuracy_score(y_test, y_predection)
lr_conf_matrix = confusion_matrix(y_test, y_predection)
lr_classification_rep = classification_report(y_test, y_predection)


# In[40]:


print(f'lr_Accuracy: {lr_accuracy:.2f}')
print('\nlr_Confusion Matrix:')
print(lr_conf_matrix)
print('\nlr_Classification Report:')
print(lr_classification_rep)


# ****Checking for the over and under fiting ****

# In[41]:


print("Training Score:",lr.score(X_train,y_train)*100,'%')
print("Testing Score:",lr.score(X_test,y_test)*100,'%')


# # **SVM Model**

# In[42]:


svm = SVC(kernel = 'linear', random_state = 20)
svm.fit(X_train, y_train)


# In[43]:


svm_y_predection=svm.predict(X_test)


# In[44]:


svm_accuracy = accuracy_score(y_test, svm_y_predection)
svm_conf_matrix = confusion_matrix(y_test, svm_y_predection)
svm_classification_rep = classification_report(y_test, svm_y_predection)


# In[45]:


print(f'svm_Accuracy: {svm_accuracy:.2f}')
print('\nsvm_Confusion Matrix:')
print(svm_conf_matrix)
print('\nsvm_Classification Report:')
print(svm_classification_rep)


# ****Checking for the over and under fiting ****

# In[46]:


print("Training Score:",svm.score(X_train,y_train)*100,'%')
print("Testing Score:",svm.score(X_test,y_test)*100,'%')


# # **DecisionTree Model**

# In[57]:


# Define the parameter grid to search
param_grid = {
    'max_depth': [3, 5, 10, None],  # depths to consider
    'min_samples_leaf': [1, 2, 4, 6] } # minimum number of samples required at a leaf node


# In[58]:


decision_tree_model = GridSearchCV(DecisionTreeClassifier(), param_grid=param_grid, cv=5)


# In[59]:


decision_tree_model.fit(X_train, y_train)


# In[60]:


y_pred_dt=decision_tree_model.predict(X_test)


# In[61]:


dt_accuracy = accuracy_score(y_test, y_pred_dt)
dt_conf_matrix = confusion_matrix(y_test, y_pred_dt)
dt_classification_rep = classification_report(y_test, y_pred_dt)


# In[62]:


print(f'dt_Accuracy: {dt_accuracy:.2f}')
print('\ndt_Confusion Matrix:')
print(dt_conf_matrix)
print('\ndt_Classification Report:')


# **Checking for the over and under fiting**

# In[63]:


print("Training Score:",decision_tree_model.score(X_train,y_train)*100,'%')
print("Testing Score:",decision_tree_model.score(X_test,y_test)*100,'%')


# # **RandomForest Model**

# In[12]:


# Define the parameter grid to search
param_grid_ = {
    'n_estimators': [10, 50, 300, 200],  # Number of trees in the forest
    'max_depth': [None, 10, 20, 30],  # Maximum depth of the tree
    'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4]  # Minimum number of samples required to be at a leaf node
}

# Initialize the grid search model
random_forest_model = GridSearchCV(RandomForestClassifier(), param_grid=param_grid_, cv=2, n_jobs=-1)


# In[13]:


random_forest_model = RandomForestClassifier()
random_forest_model.fit(X_train,y_train)


# In[14]:


y_pred_rf=random_forest_model.predict(X_test)


# In[15]:


rf_accuracy = accuracy_score(y_test, y_pred_rf)
rf_conf_matrix = confusion_matrix(y_test, y_pred_rf)
rf_classification_rep = classification_report(y_test, y_pred_rf)


# In[16]:


print(f'rf_Accuracy: {rf_accuracy:.3f}')
print('\nrf_Confusion Matrix:')
print(rf_conf_matrix)
print('\nrf_Classification Report:')
print(rf_classification_rep)


# **Checking for the over and under fiting**

# In[17]:


print("Training Score:",random_forest_model.score(X_train,y_train)*100,'%')
print("Testing Score:",random_forest_model.score(X_test,y_test)*100,'%')


# # **Making a Predictive System**

# In[19]:


input_data = (1,80.0,0,1,1,25.19,6.6,140)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = random_forest_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')


# # **Saving the trained model**

# In[20]:


import pickle


# In[21]:


filename = 'diabetes_model.sav'
pickle.dump(random_forest_model, open(filename, 'wb'))


# In[22]:


# loading the saved model
loaded_model = pickle.load(open('diabetes_model.sav', 'rb'))


# In[23]:


input_data = (1,80.0,0,1,1,25.19,6.6,140)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')


# In[24]:


for column in X.columns:
  print(column)


# In[ ]:


from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load your trained model (make sure the path is accessible from your Jupyter Notebook)
#model = pickle.load(open('rf.pkl', 'rb'))
@app.route('/')
def home():
    return "Welcome to the Diabetes Prediction API"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    print(data["gender"])
    features = [data['gender'], data['age'], data['hypertension'], data['heart_disease'],
                data['smoking_history'], data['bmi'], data['HbA1c_level'], data['blood_glucose_level']]
    prediction = rf.predict([features])
    return jsonify({'diabetes_prediction': int(prediction[0])})

from werkzeug.serving import run_simple
run_simple('localhost', 8000, app)


# In[ ]:





# In[ ]:




