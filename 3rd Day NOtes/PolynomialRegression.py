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


#Split the dataset into the Training & Test Set
#Feature Scaling
#No Need to train as dataset is small & no need to do feature scaling also
#Fitting linear regression to the dataset


from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

#Fitting Polynomial regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly=poly_reg.fit_transform(X)


#Now Create another linear regression using the above X_poly
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly,y)

#Visualize the linear regression results
plt.scatter(X,y, color = 'red' )
plt.plot(X,lin_reg.predict(X),color = 'blue')
plt.title('Truth or Bluff by Interviewee (Linear Regression')
plt.xlabel('Positional Level')
plt.ylabel('Salary')
plt.show()


#Visualize the Polynomial regression results for more data
#Create 99 data points from min to max val of X and increment as 0.1
X_grid = np.arange(min(X), max(X),0.1)
X_grid_matrix=X_grid.reshape(len(X_grid),1)
plt.scatter(X,y, color = 'red' )
plt.plot(X_grid_matrix,lin_reg_2.predict(poly_reg.fit_transform(X_grid_matrix)),color = 'blue')
plt.title('Truth or Bluff by Interviewee (Polynomial regression')
plt.xlabel('Positional Level')
plt.ylabel('Salary')
plt.show()

#Predict the results with Linear Regression
#Emp is telling his position level is 6.5 & telling salary level around 150K
#Lets check whether employee is telling truth or bluff
#Instead of entering the entire X since we want to know only for 6.5 level
lin_reg.predict(6.5)

#Check this using Polynomial Regression
lin_reg_2.predict(poly_reg.fit_transform(6.5))

#Based on the above result we can conclude that employee told correct salary
