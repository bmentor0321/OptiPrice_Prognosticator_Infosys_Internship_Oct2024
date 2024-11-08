import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import numpy as np

# Paths
from constants import RAW_DIR, RESULTS_DIR

# Load the data
data = pd.read_csv(os.path.join(RAW_DIR, 'dynamic_pricing.csv'))

# Average rating class (good, bad) for average ratings
data["Average_Ratings_Class"] = np.where(data["Average_Ratings"] > 4.2, "Good", "Bad")

# Take all features as independent variables except the other categorical columns
X = data.drop(["Historical_Cost_of_Ride", "Location_Category", "Customer_Loyalty_Status", "Time_of_Booking", "Average_Ratings_Class"], axis=1)
y = data["Historical_Cost_of_Ride"]

#One-hot encoding for 'Vehicle_Type'
X_encoded = pd.get_dummies(X, columns=['Vehicle_Type'])

# Separate numerical features for scaling
numerical_columns = ["Number_of_Riders", "Number_of_Drivers", "Number_of_Past_Rides", "Expected_Ride_Duration"]
scaler = StandardScaler()
X_encoded[numerical_columns] = scaler.fit_transform(X_encoded[numerical_columns])

# Split the data into train, validation, and testing datasets
X_train, X_temp, y_train, y_temp = train_test_split(X_encoded, y, test_size=0.3, random_state=42)
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
test_results.to_csv(os.path.join(RESULTS_DIR, 'test_results_with_error_3.4.csv'), index=False)