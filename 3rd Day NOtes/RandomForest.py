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

#Fitting the randomForest model to the dataset

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 135,
                                  random_state = 500)
regressor.fit(X,y)


#Predict a new result
y_pred = regressor.predict(6.5)
print(y_pred)

#Compared to 170 trees 135 trees gives better results
#Both of these trees results are better than Decision Tree result

#Visualize the RandomForest results for higher resolution and smooth curve
#Create 99 data points from min to max val of X and increment as 0.1
X_grid = np.arange(min(X), max(X), 0.001)

#Below code converts this as Matrix
X_grid_matrix=X_grid.reshape(len(X_grid),1)
plt.scatter(X,y,color='red')

#We need to plot original data as red dots
#Actually we need to plot prediction based on Polynomial regression


plt.plot(X_grid_matrix,regressor.predict(X_grid_matrix),color = 'blue')
plt.title('Truth or Bluff by Interviewee (RandomForest Model)')
plt.xlabel('Positional Level')
plt.ylabel('Salary')
plt.show()

