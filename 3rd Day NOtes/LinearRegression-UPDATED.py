#Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Import Dataset
dataset = pd.read_csv("Salary_data-TG.csv")
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values


#Split the dataset for Training and Test
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1,random_state=0)
print (X_test)
print (y_test)

#Buidling Linear Regression Model
from sklearn.linear_model import LinearRegression
regressor  = LinearRegression()
#Training the regressor object
regressor.fit(X_train, y_train)

#Predict the model
y_pred = regressor.predict(X_test)

#Visualize the train set results
plt.scatter(X_train, y_train, color = 'red' )
plt.plot(X_train,regressor.predict(X_train),color = 'blue')
plt.title('Experience Vs Title (training set)')
plt.xlabel('Years of Exp')
plt.ylabel('Salary')
plt.show()

#Visualize the test set results
plt.scatter(X_test, y_test, color = 'red' )
plt.plot(X_test,regressor.predict(X_test),color = 'blue')
plt.title('Experience Vs Title (Test set)')
plt.xlabel('Years of Exp')
plt.ylabel('Salary')
plt.show()

