# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 15:41:41 2017

@author: training
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Import Dataset
dataset = pd.read_csv("Position_Salaries-TG1.csv")
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

#Split the dataset into the Training & Test set
#Feature Scaling
#No need to train as dataset is small and no need to do feature scaling

#Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

#Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 10)
X_poly = poly_reg.fit_transform(X)

#Now create another Linear regression using the above X_poly
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly,y)

#Visualising the Linear regression results
plt.scatter(X,y,color='red')
plt.plot(X,lin_reg.predict(X),color='blue')
plt.title('Truth or Bluff by interviewee(Linear regression)')
plt.xlabel('Positional Level')
plt.ylabel('Salary')
plt.show()

#Visualising the Plynomail regression results for more data
#CReate 99 data points from min to max val of X and increment as 0.1
X_grid = np.arange(min(X),max(X),0.1)
X_grid_matrix = X_grid.reshape(len(X_grid),1)
plt.scatter(X,y,color='red') # we need to plot original data as red dots
plt.plot(X_grid_matrix,lin_reg_2.predict(poly_reg.fit_transform(X_grid_matrix)))
plt.title('Truth or Bluff by interviewee(Polynomial regression')
plt.xlabel('Positional Level')
plt.ylabel('Salary')
plt.show()

# <jp - to increase accuracy of the model, you could increase degree to more than 2 in the code {poly_reg = PolynomialFeatures(degree = 2)} >


#Predict the result with Linear Regresion
#Employee is telling that his position level is 6.5 & telling salary around 150k
#lets check whether Employee is telling Truth or Bluff
#INstead of entering the entire X soince we want to know only for level 6.5
lin_reg.predict(6.5)

#check this using Poynomial Regression
lin_reg_2.predict(poly_reg.fit_transform(6.5))
#based on the above result as could find if employee told truth or Bluff