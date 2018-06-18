# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 17:03:08 2017

@author: training
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Import Dataset
dataset = pd.read_csv("Salary_Data-TG.csv")
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values

#Split the dataset into the Training and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)

#Linear Regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
#Training the regressor object with the Training Data
regressor.fit(X_train, y_train)

#As model is already trained predict it based on that
y_pred = regressor.predict(X_test)

#Visualize the trainingt set results
plt.scatter(X_train, y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Experience vs Title (Training Set )')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#Visualize the trainingt set results
plt.scatter(X_test, y_test,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Experience vs Title (Training Set )')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()