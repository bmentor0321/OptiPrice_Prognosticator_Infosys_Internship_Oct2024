# Import necessary libraries
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle
from Utils.constants import RAW_DIR, file_name, RESULTS_DIR, output_file_name, processed_data_dir
from Utils.utils import get_mean

# Read the data
df = data = pd.read_csv(os.path.join(RAW_DIR, file_name))

# Select numerical columns for features (X) and specify the target variable (Y)
X = data.select_dtypes(include='number').drop(columns='Historical_Cost_of_Ride', errors='ignore')
Y = data['Historical_Cost_of_Ride']

# Split the data into training, validation, and test sets
X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.4, random_state=42)
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = LinearRegression()
model.fit(X_train_scaled, Y_train)

# Make predictions
Y_train_pred = model.predict(X_train_scaled)
Y_val_pred = model.predict(X_val_scaled)
Y_test_pred = model.predict(X_test_scaled)

# Calculate RMSE
train_rmse = np.sqrt(mean_squared_error(Y_train, Y_train_pred))
val_rmse = np.sqrt(mean_squared_error(Y_val, Y_val_pred))
test_rmse = np.sqrt(mean_squared_error(Y_test, Y_test_pred))

# Print RMSE values
print("Train RMSE:", train_rmse)
print("Validation RMSE:", val_rmse)
print("Test RMSE:", test_rmse)

# Save the test results with actual, predicted values, and errors
results = X_test.copy()
results['Historical_Cost_of_Ride'] = Y_test
results['Predicted_Cost_of_Ride'] = Y_test_pred
results['Error'] = results['Historical_Cost_of_Ride'] - results['Predicted_Cost_of_Ride']

# Ensure the results directory exists
RESULTS_DIR = r'D:\Rishi\Data\Results\output_files'
os.makedirs(RESULTS_DIR, exist_ok=True)

# Define the full path for the output file
output_file_path = os.path.join(RESULTS_DIR, 'model_2_result.csv')

# Save results to the specified directory
results.to_csv(output_file_path, index=False)

#  Define results directory
RESULTS_DIR = r'D:\Rishi\Data\Results\result_files'
# Initialize metrics DataFrame
df_metrics = pd.DataFrame(columns=['Models', "Model Description", "Train error", "Val error", "Test error"])

# Add the current model's metrics to the DataFrame
df_metrics.loc[len(df_metrics)] = ["Model 2", "All Numerical Features + Standard Scaler Introduced", train_rmse, val_rmse, test_rmse]

# Check if comparison.csv already exists in RESULTS_DIR, then append; otherwise, create a new file
comparison_file_path = os.path.join(RESULTS_DIR, "comparison.csv")
if os.path.exists(comparison_file_path):
    # Read existing metrics file
    df_metrics_ = pd.read_csv(comparison_file_path)
    
    # Concatenate the new metrics to the existing ones
    df_metrics_ = pd.concat([df_metrics_, df_metrics], ignore_index=True)
else:
    # If the file doesn't exist, start with the current metrics
    df_metrics_ = df_metrics

# Save the updated metrics back to comparison.csv
df_metrics_.to_csv(comparison_file_path, index=False)