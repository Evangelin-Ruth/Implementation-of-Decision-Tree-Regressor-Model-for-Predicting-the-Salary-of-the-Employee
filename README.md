# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
```
1.Import the required libraries.
2.Upload the csv file and read the dataset.
3.Check for any null values using the isnull() function.
4.From sklearn.tree inport DecisionTreeRegressor.
5.Import metrics and calculate the Mean squared error.
6.Apply metrics to the dataset, and predict the output.
```
## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Evangelin.S
RegisterNumber:  212221230025
*/

import pandas as pd
data = pd.read_csv("Salary.csv")
data.head()
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["Position"] = le.fit_transform(data["Position"])
data.head()
x = data[["Position","Level"]]
y = data["Salary"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred = dt.predict(x_test)
from sklearn import metrics
mse = metrics.mean_squared_error(y_test,y_pred)
mse
r2 = metrics.r2_score(y_test,y_pred)
r2
dt.predict([[5,6]])

```

## Output:
## Data Head:
![image](https://user-images.githubusercontent.com/94219798/172993169-50b25e80-bc98-47ef-8d1d-06cc2367c9c4.png)

## Data Info:
![image](https://user-images.githubusercontent.com/94219798/172993202-568c8009-4c23-4e2f-9a8b-9d86a365a344.png)

## Data Head after applying LabelEncoder():
![image](https://user-images.githubusercontent.com/94219798/172993236-5bf144fb-024e-4d48-8170-9224c4b78f69.png)

## MSE:
![image](https://user-images.githubusercontent.com/94219798/172993284-10c89ea3-f9ac-4fbc-aacb-c582b9fb241b.png)

## r2:
![image](https://user-images.githubusercontent.com/94219798/172993306-3264f824-455b-45dd-a61b-8428a137c2fc.png)

## Data Prediction:
![image](https://user-images.githubusercontent.com/94219798/172993339-daa5aefa-8f18-4add-bab0-3ff91be1f0f0.png)



## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
