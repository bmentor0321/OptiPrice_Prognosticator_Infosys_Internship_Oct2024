import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from Utils.constants import RAW_DIR, file_name, RESULTS_DIR, output_file_name, processed_data_dir
from Utils.utils import get_mean

# Read the data
df = pd.read_csv(os.path.join(RAW_DIR, file_name))

# Select numerical columns for features (X) and specify the target variable (Y)
X = df.select_dtypes(include='number').drop(columns='Historical_Cost_of_Ride', errors='ignore')
Y = df['Historical_Cost_of_Ride']

# Split the data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LinearRegression()
model.fit(X_train_scaled, Y_train)

# Make predictions
Y_train_pred = model.predict(X_train_scaled)
Y_test_pred = model.predict(X_test_scaled)

# Calculate metrics
train_metrics = {
    'MSE': mean_squared_error(Y_train, Y_train_pred),
    'RMSE': np.sqrt(mean_squared_error(Y_train, Y_train_pred)),
    'MAPE': mean_absolute_percentage_error(Y_train, Y_train_pred)
}

test_metrics = {
    'MSE': mean_squared_error(Y_test, Y_test_pred),
    'RMSE': np.sqrt(mean_squared_error(Y_test, Y_test_pred)),
    'MAPE': mean_absolute_percentage_error(Y_test, Y_test_pred)
}

# Save test results with actual, predicted values, and errors
results = X_test.copy()
results['Historical_Cost_of_Ride'] = Y_test
results['Predicted_Cost_of_Ride'] = Y_test_pred
results['Error'] = results['Historical_Cost_of_Ride'] - results['Predicted_Cost_of_Ride']

# Calculate additional error metrics
results['Absolute_Error'] = abs(results['Error'])
results['Percentage_Error'] = (results['Error'] / results['Historical_Cost_of_Ride']) * 100

# Ensure the results directory exists
RESULTS_DIR = r'D:\Rishi\Data\Results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# Define the full path for the output file
output_file_path = os.path.join(RESULTS_DIR, 'model_1_result.csv')

# Save results to the specified directory
results.to_csv(output_file_path, index=False)