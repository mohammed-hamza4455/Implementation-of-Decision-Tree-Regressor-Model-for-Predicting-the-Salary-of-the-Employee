# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the libraries and read the data frame using pandas.
2.Calculate the null values present in the dataset and apply label encoder.
3.Determine test and training data set and apply decison tree regression in dataset.
4.Calculate Mean square error,data prediction and r2.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: MOHAMMED HAMZA M
RegisterNumber:  212224230167
*/
```
```py
import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()

data.info

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()

x=data[["Position","Level"]]
y=data[["Salary"]]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])
```
## Output:
## DATA HEAD:
![WhatsApp Image 2025-04-21 at 10 33 04_bb9555ac](https://github.com/user-attachments/assets/dc658d79-ec61-4cc2-b820-8d5ef677a717)

## DATA INFO:
![WhatsApp Image 2025-04-21 at 10 33 34_0b8a4ddf](https://github.com/user-attachments/assets/69e5a57a-73d4-434e-8521-1a473b16fd82)

## ISNULL() AND SUM():
![WhatsApp Image 2025-04-21 at 10 34 28_d31690e7](https://github.com/user-attachments/assets/439f785b-a4f0-4d03-a400-12a67c29c18f)

## DATA HEAD FOR SALARY:
![WhatsApp Image 2025-04-21 at 10 34 54_8da500d5](https://github.com/user-attachments/assets/0a78b483-eb56-4cdc-8c50-0a5e7d3d2773)

## MEAN SQUARED ERROR:
![WhatsApp Image 2025-04-21 at 10 35 27_0a58574d](https://github.com/user-attachments/assets/ed8c2366-1684-43c9-b9fc-a5c5f9d06b76)

## R2 VALUE:
![WhatsApp Image 2025-04-21 at 10 35 46_9a29aa32](https://github.com/user-attachments/assets/1e584499-ad1e-4a17-8451-4ab9cf3454c1)

## DATA PREDICTION:
![WhatsApp Image 2025-04-21 at 10 36 06_0e540572](https://github.com/user-attachments/assets/ddedeb34-f7b4-4829-8a5f-1dad0b515e44)



## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.

