# Import necessary libraries
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from Utils.constants import RAW_DIR ,RESULT_FILES_DIR, RAW_FILE_PATH, OUTPUT_FILES_DIR, LOYALTY_DISCOUNTS, BASE_FARE,PEAK_HOURS_MULTIPLIER, TRAFFIC_MULTIPLIER, PAST_RIDES_DISCOUNT, DURATION_MULTIPLIER, DURATION_THRESHOLD, HISTORICAL_COST_ADJUSTMENT_FACTOR, DISTANCE_THRESHOLD,output_file_name

# Read the data
df = pd.read_csv(os.path.join(RAW_DIR, RAW_FILE_PATH))

# Select numerical columns for features (X) and specify the target variable (Y)
X = df.select_dtypes(include='number').drop(columns='Historical_Cost_of_Ride', errors='ignore')
Y = df['Historical_Cost_of_Ride']

# Split the data into training, validation, and test sets
X_train_full, X_test, Y_train_full, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
X_train, X_val, Y_train, Y_val = train_test_split(X_train_full, Y_train_full, test_size=0.125, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LinearRegression()
model.fit(X_train_scaled, Y_train)

# Make predictions
Y_train_pred = model.predict(X_train_scaled)
Y_val_pred = model.predict(X_val_scaled)
Y_test_pred = model.predict(X_test_scaled)

# Calculate RMSE for train, validation, and test sets
train_rmse = np.sqrt(mean_squared_error(Y_train, Y_train_pred))
val_rmse = np.sqrt(mean_squared_error(Y_val, Y_val_pred))
test_rmse = np.sqrt(mean_squared_error(Y_test, Y_test_pred))

# Initialize DataFrame with the metrics columns
df_metrics = pd.DataFrame(columns=['Models', "Model Description", "Train error", "Val error", "Test error"])

# Add the current model's metrics with calculated RMSE values
df_metrics.loc[len(df_metrics)] = ["Model 1", "All Numerical Features", train_rmse, val_rmse, test_rmse]

# Save or append to comparison.csv in RESULTS_DIR
comparison_file_path = os.path.join(RESULT_FILES_DIR, "comparison.csv")
if os.path.exists(comparison_file_path):
    # Read existing metrics file
    df_existing_metrics = pd.read_csv(comparison_file_path)
    # Append new metrics and avoid re-indexing
    df_combined_metrics = pd.concat([df_existing_metrics, df_metrics], ignore_index=True)
else:
    df_combined_metrics = df_metrics

# Save the combined metrics to comparison.csv
df_combined_metrics.to_csv(comparison_file_path, index=False)

# Save test results with actual, predicted values, and errors
results = X_test.copy()
results['Historical_Cost_of_Ride'] = Y_test
results['Predicted_Cost_of_Ride'] = Y_test_pred
results['Error'] = results['Historical_Cost_of_Ride'] - results['Predicted_Cost_of_Ride']

# Ensure the output_files directory exists
output_files_dir = os.path.join(output_file_name, 'output_files')
os.makedirs(output_files_dir, exist_ok=True)

# Define the full path for the output file
output_file_path = os.path.join(output_files_dir, 'model_1_result.csv')

# Save results to the specified directory
results.to_csv(output_file_path, index=False)