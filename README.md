# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Data Preparation
Load the dataset.
Convert categorical output (Placed, Not Placed) into numerical values (1, 0).
Select input features and normalize them using Standard Scaler.
Add a bias (intercept) term to the feature matrix. 
2. Model Initialization
Initialize the weight vector (theta) with zeros.
Define the sigmoid function to map values between 0 and 1.
Define the cost (loss) function using logistic regression formula.
3. For a fixed number of iterations:
Compute predicted values using the sigmoid function.
Calculate the gradient of the cost function.
Update the weights using the learning rate.
Store the cost value for analysis.
4.Predict class labels using a threshold (0.5).
Compare predicted values with actual values.
Calculate and display model accuracy.
Plot cost vs iterations to visualize convergence. 
```
## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by:ASWINI D 
RegisterNumber: 25018420 
*/
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
data = pd.read_csv(r"C:\Users\acer\Downloads\Placement_Data (2).csv")
data['status'] = data['status'].map({'Placed': 1, 'Not Placed': 0})
X = data[['ssc_p', 'mba_p']].values
y = data['status'].values
scaler = StandardScaler()
X = scaler.fit_transform(X)
m = len(y)
X = np.c_[np.ones(m), X]
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def cost_function(X, y, theta):
    h = sigmoid(X @ theta)
    return (-1/m) * np.sum(y*np.log(h) + (1-y)*np.log(1-h))
theta = np.zeros(X.shape[1])
alpha = 0.1
cost_history = []

for i in range(500):
    z = X @ theta
    h = sigmoid(z)
    gradient = (1/m) * X.T @ (h - y)
    theta = theta - alpha * gradient
    
    cost = cost_function(X, y, theta)
    cost_history.append(cost)
y_pred = (sigmoid(X @ theta) >= 0.5).astype(int)
accuracy = np.mean(y_pred == y) * 100
print("Weights:", theta)
print("Accuracy:", accuracy, "%")
plt.figure()
plt.plot(cost_history)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Logistic Regression using Gradient Descent")
plt.show()
```

## Output:
<img width="857" height="637" alt="image" src="https://github.com/user-attachments/assets/9ce42f7f-0ce2-4de7-9051-badb04f7c325" />


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

