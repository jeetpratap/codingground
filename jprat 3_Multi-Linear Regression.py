# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 15:18:34 2017

@author: training
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Import Dataset
dataset = pd.read_csv("50_Startups-TG.csv")
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values

#Encoding Categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,3] = labelencoder_X.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features=[3])

X = onehotencoder.fit_transform(X).toarray()

#To avoid dummy variable trip
X = X[:,1:]

#Split the dataset into the Training and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)

#Fitting Multiple Linear Regression to the Training set
from sklearn_Linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#Predicting the result
y_pred = regressor.predict(X_test)

#Now we are going to build the model using Backward Elimination Model
import statsmodels.formula.api as sm
X = np.append(np.ones(shape=(50,1)).astype(int),values=X,axis=1)

#Now we are going to eliminate the independent variable that is not greater
#than significant Level. Sl is 0.05 (i.e.5%)
X_opt = X[:,[0,1,2,3,4,5]]

#OLS is simple ordinary Least Square  model. First parameter is dependent
#after running summary check the p-value as specified above

regressor_OLS = sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,2,3,4,5]]
regressor_OLS = sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,3,4,5]]
regressor_OLS = sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,3,5]]
regressor_OLS = sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,3]]
regressor_OLS = sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()


