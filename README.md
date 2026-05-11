# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import required libraries such as Pandas and sklearn.
2. Load dataset and split into features (X) and target (y).
3. Divide dataset into training and testing sets.
4. Create Decision Tree Regressor model.
5. Train model and predict salary values.
6. Evaluate model using MSE, MAE, and R² score and visualize tree.

## Program:
# Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
# Developed by: VIJIYALAKSHMI A
# RegisterNumber: 212225240185 
```
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Dataset
data = {
    'Experience': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Salary': [25000, 30000, 35000, 45000, 55000, 65000, 75000, 85000, 95000, 110000]
}

df = pd.DataFrame(data)

# Features & Target
X = df[['Experience']]
y = df['Salary']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# -------- Evaluation Metrics --------
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)
print("R2 Score:", r2)

# -------- Decision Tree Diagram --------
plt.figure(figsize=(12,6))
plot_tree(model,
          feature_names=['Experience'],
          filled=True,
          rounded=True)
plt.title("Decision Tree Regressor - Salary Prediction")
plt.show()
```

## Output:
<img width="1256" height="698" alt="Screenshot 2026-05-11 191016" src="https://github.com/user-attachments/assets/0437cd6d-9c28-4af9-bf96-c05fbbeea5ba" />

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
