import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Load the data from the specified file path
file_path = r"C:\Users\chide\OneDrive\Desktop\infosys\dynamic_pricing.csv"
data = pd.read_csv(file_path, encoding='utf-8')

# Separate the target variable (Y) and features (X)
X = data.drop(columns=['Historical_Cost_of_Ride'])
y = data['Historical_Cost_of_Ride']

# One-hot encode categorical variables
X_encoded = pd.get_dummies(X, columns=['Location_Category', 'Customer_Loyalty_Status', 
                                       'Time_of_Booking', 'Vehicle_Type'], drop_first=True)

# Split data into train, validation, and test sets (70% train, 15% validation, 15% test)
X_train, X_temp, y_train, y_temp = train_test_split(X_encoded, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Train a Random Forest Regressor model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Validate the model on the validation set
y_val_pred = model.predict(X_val)
val_mae = mean_absolute_error(y_val, y_val_pred)
val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))

# Test the model on the test set
y_test_pred = model.predict(X_test)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

# Calculate individual errors (residuals) for the test set
test_errors = y_test - y_test_pred

# Create a DataFrame with actual Y, predicted Y, and error for the test set
test_results = X_test.copy()
test_results['Y_Actual'] = y_test
test_results['Y_Predicted'] = y_test_pred
test_results['Error'] = test_errors

# Save the test dataset with Y actual, Y predicted, and error into a CSV file
output_path = r"C:\Users\chide\OneDrive\Desktop\infosys\test_results_with_errors.csv"
test_results.to_csv(output_path, index=False)

# Display validation and test error metrics
print("Validation MAE:", val_mae)
print("Validation RMSE:", val_rmse)
print("Test MAE:", test_mae)
print("Test RMSE:", test_rmse)

# Display first few rows of the test results DataFrame
print(test_results.head())
