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
dataset = pd.read_csv("Position_Salaries-TG.csv")
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values


#Feature Scaling Not Required

#Fitting the regression model to the dataset

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X,y)


#Predict a new result
y_pred = regressor.predict(6.5)


#Visualize the Decision Tree results
#This is the non continnous regression that we are seeing

plt.scatter(X,y, color = 'red' )
plt.plot(X,regressor.predict(X),color = 'blue')
plt.title('Truth or Bluff by Interviewee (Decision Tree)')
plt.xlabel('Positional Level')
plt.ylabel('Salary')
plt.show()

