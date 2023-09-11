# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.import the needed packages.

2.Assigning hours to x and scores to y.

3.Plot the scatter plot.

4.Use mse,rmse,mae formula to find the values.


## Program:

```python
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Bergin.S
RegisterNumber: 212222040025
# IMPORT REQUIRED PACKAGE
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt
dataset=pd.read_csv('student_scores.csv')
print(dataset)
# READ CSV FILES
dataset=pd.read_csv('student_scores.csv')
print(dataset.head())
print(dataset.tail())
# COMPARE DATASET
x=dataset.iloc[:,:-1].values
print(x)
y=dataset.iloc[:,1].values
print(y)
# PRINT PREDICTED VALUE
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
y_pred = reg.predict(x_test)
print(y_pred)
print(y_test)
# GRAPH PLOT FOR TRAINING SET
plt.scatter(x_train,y_train,color='purple')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
# GRAPH PLOT FOR TESTING SET
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
# PRINT THE ERROR
mse=mean_absolute_error(y_test,y_pred)
print('Mean Square Error = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('Mean Absolute Error = ',mae)
rmse=np.sqrt(mse)
print("Root Mean Square Error = ",rmse)
*/
```

## Output:

To read csv file

![265652657-9c4c9092-d20a-4bf4-aeb3-c0b5c3d5c913](https://github.com/bergin1312/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119404594/918a6047-5de4-4122-81a5-be8ee4183cfa)

To Read Head and Tail Files

![265652888-c1ac175f-b39c-4ed3-9236-d49eed3da214](https://github.com/bergin1312/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119404594/670cad7c-eeac-4c9a-aaf4-39eac9d4c765)

Compare Dataset

![265653044-d6551083-db41-4e0e-8f48-52da085630e8](https://github.com/bergin1312/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119404594/10ec6aac-aadc-4168-a5fe-176f514cf34c)

Predicted Value

![265653205-f01d961d-274e-47e5-96a1-6e32371fdaa0](https://github.com/bergin1312/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119404594/f1bd0ea8-f64f-4122-971f-cb7afb45b9e7)

Graph For Training Set

![265653364-e5aef6c8-dc0c-421b-8c90-8a1166715f8d](https://github.com/bergin1312/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119404594/8ec20bda-9284-4449-b629-2a5e9489a8ec)

Graph For Testing Set

![265705610-876779cd-bc6c-4a3d-91fb-bdcc98602f53](https://github.com/bergin1312/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119404594/661dea01-c64d-47c2-a1be-de35f1f54a9d)

Error

![265705680-d257abd8-a7a5-4f1b-a4c2-0bce8db497b4](https://github.com/bergin1312/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119404594/a2794f42-9663-4344-ab28-279e7505d29b)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
