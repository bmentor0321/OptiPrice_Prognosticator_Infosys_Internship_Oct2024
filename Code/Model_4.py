# Import necessary libraries
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from Utils.constants import RAW_DIR, file_name, RESULTS_DIR
from Utils.utils import get_mean

# Load the data
NEW_DATAPATH = r'D:\Rishi\Data\Raw\dynamic_pricing.csv'
try:
    df = pd.read_csv(NEW_DATAPATH)
    df.columns = df.columns.str.strip()  # Remove any leading/trailing whitespace in column names
    print("Data loaded successfully.")
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit()

# Check if 'Historical_Cost_of_Ride' is in the columns
if 'Historical_Cost_of_Ride' not in df.columns:
    print("Error: 'Historical_Cost_of_Ride' column is missing from the dataset.")
    print("Available columns:", df.columns)
    exit()

# Define a function to categorize ratings
def categorize_rating(rating):
    if rating >= 4.0:
        return 'Excellent'
    elif rating >= 3.0:
        return 'Good'
    else:
        return 'Poor'

# Add new column for rating categories
df['Rating_Category'] = df['Average_Ratings'].apply(categorize_rating)

# Remove 'Poor' ratings
df = df[df['Rating_Category'] != 'Poor']

# Combine specified categorical features into a single feature using one-hot encoding
combined_features = ['Customer_Loyalty_Status', 'Location_Category', 'Vehicle_Type', 'Time_of_Booking']
df_encoded = pd.get_dummies(df, columns=combined_features)

# Define 'Adjusted_Historical_Cost_of_Ride' as the target variable (y)
df_encoded['Adjusted_Historical_Cost_of_Ride'] = df_encoded['Historical_Cost_of_Ride']

# Drop unnecessary columns
df_encoded.drop(['Historical_Cost_of_Ride', 'Average_Ratings', 'Rating_Category'], axis=1, inplace=True, errors='ignore')

# Prepare features (X) and target (y)
X = df_encoded.drop('Adjusted_Historical_Cost_of_Ride', axis=1)
y = df_encoded['Adjusted_Historical_Cost_of_Ride']

# Scale the numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into train, validation, and test sets (60% train, 20% validation, 20% test)
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Train linear regression model on the training set
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the train, validation, and test sets
y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)

# Calculate RMSE for each dataset
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

# Print RMSE values
print(f"Train RMSE: {train_rmse}")
print(f"Validation RMSE: {val_rmse}")
print(f"Test RMSE: {test_rmse}")

# Initialize DataFrame to save model comparison metrics
df_metrics = pd.DataFrame(columns=['Models', "Model Description", "Train error", "Val error", "Test error"])

# Add the current model's metrics
df_metrics.loc[len(df_metrics)] = ["Model 4", "All Numerical Features + Standard Scaler Introduced + Categorical Features One Hot Encoded + Adjusted Cost Of Ride", train_rmse, val_rmse, test_rmse]

# Define the path for comparison.csv in RESULTS_DIR
result_files = r'D:\Rishi\Data\Results\result_files'
comparison_file_path = os.path.join(result_files, "comparison.csv")

# Save or append the comparison.csv file
if os.path.exists(comparison_file_path):
    # If file exists, load existing data and append the new metrics
    df_existing_metrics = pd.read_csv(comparison_file_path)
    df_combined_metrics = pd.concat([df_existing_metrics, df_metrics], ignore_index=True)
else:
    # If file doesn't exist, save df_metrics as the new file
    df_combined_metrics = df_metrics

# Save combined metrics to comparison.csv
df_combined_metrics.to_csv(comparison_file_path, index=False)

# Inverse transform to get original scaled values for numerical features
X_test_original = scaler.inverse_transform(X_test)

# Convert X_test to a DataFrame with original feature names
X_test_df = pd.DataFrame(X_test_original, columns=X.columns)

# Add prediction results to the original test data
original_test_data = X_test_df.copy()
original_test_data['Actual_Adjusted_Cost'] = y_test.values
original_test_data['Predicted_Adjusted_Cost'] = y_test_pred
original_test_data['Error'] = original_test_data['Actual_Adjusted_Cost'] - original_test_data['Predicted_Adjusted_Cost']

# Ensure the output_files directory exists
output_files_dir = os.path.join(RESULTS_DIR, 'output_files')
os.makedirs(output_files_dir, exist_ok=True)

# Define the full path for the output file
output_file_path = os.path.join(output_files_dir, 'model_4_result.csv')

# Save the test results to a CSV file
original_test_data.to_csv(output_file_path, index=False)
print(f"Results saved to {output_file_path}")
print(f"Comparison metrics saved to {comparison_file_path}")