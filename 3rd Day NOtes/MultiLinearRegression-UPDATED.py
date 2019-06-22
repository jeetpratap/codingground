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
dataset = pd.read_csv("50_Startups-TG.csv")
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values

#Encoding Categorical Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,3] = labelencoder_X.fit_transform(X[:,3])
onehotencoder=OneHotEncoder(categories='auto')

X=onehotencoder.fit_transform(X).toarray()

#To avoid dummy variable trip
X=X[:, 1:]

#Split the dataset into the Training & Test Set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.01,random_state=0)


#Fitting Multiple Linear Regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#Predict the model
y_pred = regressor.predict(X_test)

#Now we are going to build the model using Backward
import statsmodels.formula.api as sm
X=np.append(np.ones(shape=(50,1)).astype(int),values=X, axis=1)

#Now we are going to eliminate the independent variable that is not
#greater than significant level. SL is 0.05 (i.e 5%)
X_opt = X[:,[0,1,2,3,4,5]]

#OLS is Simple Ordinary Least Squares model.First parameter is dependent
#after running summary check the p value as specified above
regressor_OLS = sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()








