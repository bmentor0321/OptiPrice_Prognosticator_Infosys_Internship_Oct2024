import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

# Paths
from constants import RAW_DIR, RESULTS_DIR

# Load the data
data = pd.read_csv(os.path.join(RAW_DIR, 'dynamic_pricing.csv'))

# Feature Engineering: Demand and Supply Dynamics
data['Supply_Demand_Difference'] = data['Number_of_Drivers'] - data['Number_of_Riders']
data['d_s_ratio'] = data['Number_of_Riders'] / (data['Number_of_Drivers'] + 1e-5)

# Demand Class and Supply Class
data['demand_class'] = data['Number_of_Riders'].apply(lambda x: 'Low' if x < 10 else 'Medium' if x < 50 else 'High')
data['supply_class'] = data['Number_of_Drivers'].apply(lambda x: 'Low' if x < 10 else 'Medium' if x < 50 else 'High')

# Demand-Supply Metric
def demand_supply_metric(row):
    if row['demand_class'] == 'High' and row['supply_class'] == 'Low':
        return 'Deficit'
    elif row['demand_class'] == 'Low' and row['supply_class'] == 'High':
        return 'Surplus'
    else:
        return 'Balanced'

data['demand_supply_metric'] = data.apply(demand_supply_metric, axis=1)

# One-hot encoding for categorical columns
categorical_columns = ['Location_Category', 'Customer_Loyalty_Status', 'Time_of_Booking', 'Vehicle_Type', 'demand_class', 'supply_class', 'demand_supply_metric']
data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

# Prepare the features (X) and target (y)
X = data.drop("Historical_Cost_of_Ride", axis=1)
y = data["Historical_Cost_of_Ride"]

# Standardize numerical features
numerical_columns = ["Number_of_Riders", "Number_of_Drivers", "Number_of_Past_Rides", "Average_Ratings", "Expected_Ride_Duration", "Supply_Demand_Difference", "d_s_ratio"]
scaler = StandardScaler()
X[numerical_columns] = scaler.fit_transform(X[numerical_columns])

# Split the data into train, validation, and testing datasets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Validation check
y_val_pred = model.predict(X_val)
val_mae = mean_absolute_error(y_val, y_val_pred)
val_mse = mean_squared_error(y_val, y_val_pred)
print(f"Validation MAE: {val_mae}")
print(f"Validation MSE: {val_mse}")

# Test the model and get Y (Predicted)
y_test_pred = model.predict(X_test)

# Calculate the error between Y and Y (Predicted)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
print(f"Test MAE: {test_mae}")
print(f"Test MSE: {test_mse}")

# Save Test dataset + Y variable + Y(predicted) + Error into a CSV file
test_results = pd.DataFrame(X_test)
test_results['Historical_Cost_of_Ride'] = y_test
test_results['Predicted_Cost_of_Ride'] = y_test_pred
test_results['Error'] = test_results['Historical_Cost_of_Ride'] - test_results['Predicted_Cost_of_Ride']

# Save to CSV in Results directory
os.makedirs(RESULTS_DIR, exist_ok=True)
test_results.to_csv(os.path.join(RESULTS_DIR, 'test_results_with_error_4.csv'), index=False)
