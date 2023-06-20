# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 13:22:11 2022

@author: Fatemeh
"""
#Importing the Dependencies
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

#______________________________________________________________________________
#Data Collection and Processing

# loading the dataset to pandas DataFrame
loan_dataset = pd.read_csv('content.csv')
print(loan_dataset)

# printing the first 5 rows of the dataframe
head = loan_dataset.head()
print(head)

# number of rows and columns
shape = loan_dataset.shape
print(shape)

# statistical measures
describe = loan_dataset.describe()
print(describe)

# number of missing values in each column
missing_values = loan_dataset.isnull().sum()
print(missing_values)

# dropping the missing values
loan_dataset = loan_dataset.dropna()
print(loan_dataset)

# number of missing values in each column
after_drop = loan_dataset.isnull().sum()
print(after_drop)

# label encoding
inplace_lable = loan_dataset.replace({"Loan_Status":{'N':0,'Y':1}},inplace=True)

# printing the first 5 rows of the dataframe
lable = loan_dataset.head()
print(lable)

# Dependent column values
Dependent_values = loan_dataset['Dependents'].value_counts()
print(Dependent_values)

# replacing the value of 3+ to 4
loan_dataset = loan_dataset.replace(to_replace='3+', value=4)
print(loan_dataset)

# dependent values
Dependent_values = loan_dataset['Dependents'].value_counts()
print(Dependent_values)

#______________________________________________________________________________
#Data Visualization
# education & Loan Status
sns.countplot(x='Education',hue='Loan_Status',data=loan_dataset)

# marital status & Loan Status
sns.countplot(x='Married',hue='Loan_Status',data=loan_dataset)


#______________________________________________________________________________
# convert categorical columns to numerical values
loan_dataset.replace({'Married':{'No':0,'Yes':1},'Gender':{'Male':1,'Female':0},'Self_Employed':{'No':0,'Yes':1},
                      'Property_Area':{'Rural':0,'Semiurban':1,'Urban':2},'Education':{'Graduate':1,'Not Graduate':0}},inplace=True)


#______________________________________________________________________________
# separating the data and label
X = loan_dataset.drop(columns=['Loan_ID','Loan_Status'],axis=1)
Y = loan_dataset['Loan_Status']
print(X,Y)

#Train Test Split
X_train, X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.1,stratify=Y,random_state=2)
print(X.shape, X_train.shape, X_test.shape)

#______________________________________________________________________________
#Training the model
#Support Vector Machine Model
classifier = svm.SVC(kernel='linear')

#training the support Vector Macine model
classifier.fit(X_train,Y_train)

#______________________________________________________________________________
#Model Evaluation
# accuracy score on training data
X_train_prediction = classifier.predict(X_train)
training_data_accuray = accuracy_score(X_train_prediction,Y_train)
print('Accuracy on training data : ', training_data_accuray)

# accuracy score on training data
X_test_prediction = classifier.predict(X_test)
test_data_accuray = accuracy_score(X_test_prediction,Y_test)
print('Accuracy on test data : ', test_data_accuray)
