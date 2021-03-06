# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Import Dataset
dataset = pd.read_csv("")
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,3].values

#Identify and Fixing Missing Values
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='Nan', strategy='mode',axis=0)
imputer = imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])
print (X)

#Categorial Variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])
onehotencoder_X = OneHotEncoder(categorical_features=[0])
X = onehotencoder_X.fit_transform(X).toarray()
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
print (X)
print (y)

#Split the dataset for Training and Test
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

#Feature Scale
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
