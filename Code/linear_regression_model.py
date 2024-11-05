import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from Utils.constants import RAW_DIR, file_name, RESULTS_DIR, output_file_name, processed_data_dir
from Utils.utils import get_mean

# Ensure the processed data directory exists
os.makedirs(processed_data_dir, exist_ok=True)

# Load the data
try:
    data = pd.read_csv(os.path.join(RAW_DIR, file_name))
    print("Data loaded successfully.")
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit()

# Select numerical columns for features (X) and specify the target variable (Y)
X = data.select_dtypes(include='number').drop(columns='Historical_Cost_of_Ride', errors='ignore')
Y = data['Historical_Cost_of_Ride']

# Split the data into train, validation, and test sets
# First Split: 60% Train, 40% Temporary (to be split into validation and test)
X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.4, random_state=42)
# Second Split: 75% of Temporary for Validation, 25% for Test
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.25, random_state=42)

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train, Y_train)
print("Model training completed.")

# Evaluate model performance on the validation set
Y_val_pred = model.predict(X_val)
val_mse = mean_squared_error(Y_val, Y_val_pred)
val_mae = mean_absolute_error(Y_val, Y_val_pred)
print(f'Validation Mean Absolute Error (MAE): {val_mae}')
print(f'Validation Mean Squared Error (MSE): {val_mse}')

# Test the model and evaluate performance on test set
Y_test_pred = model.predict(X_test)
test_mse = mean_squared_error(Y_test, Y_test_pred)
test_mae = mean_absolute_error(Y_test, Y_test_pred)
print(f'Test Mean Squared Error (MSE): {test_mse}')
print(f'Test Mean Absolute Error (MAE): {test_mae}')

# Save the test results with actual, predicted values, and errors
results = X_test.copy()
results['Historical_Cost_of_Ride'] = Y_test
results['Predicted_Cost_of_Ride'] = Y_test_pred
results['Error'] = results['Historical_Cost_of_Ride'] - results['Predicted_Cost_of_Ride']

# Ensure the results directory exists
results_dir = r'D:\Rishi\Data\Results'
os.makedirs(results_dir, exist_ok=True)

# Save results to a CSV file
output_file_path = os.path.join(RESULTS_DIR, 'linear_regression_model.csv')

#Model training completed.

# Validation Mean Absolute Error (MAE): 59.02092479766246
# Validation Mean Squared Error (MSE): 5689.572527801274
# Test Mean Squared Error (MSE): 4292.288211232007
# Test Mean Absolute Error (MAE): 51.8031073536261