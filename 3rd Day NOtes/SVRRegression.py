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


#Feature Scaling

#SVR does not take care of features scaling hence we need to do that
#But simple linear regression feature scaling already there

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
x = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)


#Fitting the regression model to the dataset
#Create your own regression here
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X,y)


#Predict a new result
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(
np.array([[6.5]]))))



#Visualize the linear regression results
plt.scatter(X,y, color = 'red' )
plt.plot(X,regressor.predict(X),color = 'blue')
plt.title('Truth or Bluff by Interviewee (SVR Regression')
plt.xlabel('Positional Level')
plt.ylabel('Salary')
plt.show()

