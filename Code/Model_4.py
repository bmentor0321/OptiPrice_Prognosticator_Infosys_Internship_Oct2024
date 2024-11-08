# Import necessary libraries
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from Utils.constants import RAW_DIR, file_name, RESULTS_DIR, output_file_name, processed_data_dir
from Utils.utils import get_mean

# File paths
NEW_DATAPATH = r'D:\Rishi\Data\Raw\dynamic_pricing.csv'  
df = data = RAW_DIR = pd.read_csv(os.path.join(RAW_DIR, file_name))

# Load the data
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

# Make predictions on the test set
y_test_pred = model.predict(X_test)

# Inverse transform to get original scaled values for numerical features
X_test_original = scaler.inverse_transform(X_test)

# Convert X_test to a DataFrame with original feature names
X_test_df = pd.DataFrame(X_test_original, columns=X.columns)

# Add 1 and 0 conversion for categorical columns (optional fix if needed for boolean interpretation)
X_test_df = X_test_df.applymap(lambda x: True if x == 1 else (False if x == 0 else x))

# Extract test indices before splitting
test_indices = X_test_df.index

# Retrieve the original feature data (unscaled) from df_encoded using test indices
original_test_data = df_encoded.iloc[test_indices].copy()

# Add prediction results to the original test data
original_test_data['Actual_Adjusted_Cost'] = y_test.values
original_test_data['Predicted_Adjusted_Cost'] = y_test_pred
original_test_data['Error'] = original_test_data['Actual_Adjusted_Cost'] - original_test_data['Predicted_Adjusted_Cost']

# Ensure the results directory exists
RESULTS_DIR = r'D:\Rishi\Data\Results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# Define the output file path
output_file_path = os.path.join(RESULTS_DIR, 'model_4_result.csv')

# Save the results to a CSV file including all original test features and predictions
original_test_data.to_csv(output_file_path, index=False)