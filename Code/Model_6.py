import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from Utils.constants import RAW_DIR, file_name, RESULTS_DIR
from Utils.utils import get_mean


# Read the data
df = pd.read_csv(os.path.join(RAW_DIR, file_name))
os.makedirs(RESULTS_DIR, exist_ok=True)

# Feature Engineering
LOYALTY_DISCOUNTS = {"Regular": 0, "Silver": 0.05, "Gold": 0.10}
BASE_FARE = 50
PEAK_HOURS_MULTIPLIER = 1.5
TRAFFIC_MULTIPLIER = 1.2
PREMIUM_VEHICLE_MULTIPLIER = 2.0
PAST_RIDES_DISCOUNT = 0.01
DURATION_MULTIPLIER = 0.05
DURATION_THRESHOLD = 30
HISTORICAL_COST_ADJUSTMENT_FACTOR = 1.1

# Create categorical codes for categorical features
df['booking_category'] = pd.Categorical(df['Time_of_Booking']).codes
df['location_code'] = pd.Categorical(df['Location_Category']).codes
df['loyalty_code'] = pd.Categorical(df['Customer_Loyalty_Status']).codes
df['vehicle_code'] = pd.Categorical(df['Vehicle_Type']).codes

# Create interaction features
df['riders_to_drivers_ratio'] = df['Number_of_Riders'] / (df['Number_of_Drivers'] + 1)
df['duration_per_rider'] = df['Expected_Ride_Duration'] / df['Number_of_Riders']
df['rating_ride_interaction'] = df['Average_Ratings'] * df['Expected_Ride_Duration']

# Loyalty Discount Feature
df['Loyalty_Discount_Applied'] = df['Customer_Loyalty_Status'].map(LOYALTY_DISCOUNTS)
df['Cumulative_Past_Rides_Discount'] = (df['Number_of_Past_Rides'] // 50) * PAST_RIDES_DISCOUNT

# Traffic and Peak Hours Multipliers
df['Traffic_Condition_Adjusted_Fare'] = df['Location_Category'].apply(lambda x: TRAFFIC_MULTIPLIER if x == "Heavy" else 1)
df['Peak_Hours_Adjusted_Fare'] = df['Time_of_Booking'].apply(lambda x: PEAK_HOURS_MULTIPLIER if x in ["8-10 AM", "5-8 PM"] else 1)

# Dynamic Demand-Based Surge Multiplier
df['demand_surge_multiplier'] = df['riders_to_drivers_ratio'].apply(lambda x: 1.5 if x > 1 else 0.8 if x < 0.5 else 1)

# Ride Distance-Based Fare Adjustment
DISTANCE_THRESHOLD = 10
df['distance_adjustment'] = (df['Expected_Ride_Duration'] - DISTANCE_THRESHOLD).apply(lambda x: 0.02 * max(0, x))

# Elasticity of Demand Based on Duration
df['elasticity_fare_adjustment'] = 1 + (0.05 * (df['Expected_Ride_Duration'] > DURATION_THRESHOLD))

# Historical Cost Adjustment
df['Historical_Cost_Adjusted_Fare'] = df['Historical_Cost_of_Ride'] * HISTORICAL_COST_ADJUSTMENT_FACTOR

# Calculate total engineered fare
df['Engineered_Fare'] = (
    BASE_FARE * (1 - df['Loyalty_Discount_Applied']) *
    (1 - df['Cumulative_Past_Rides_Discount']) *
    df['Traffic_Condition_Adjusted_Fare'] *
    df['Peak_Hours_Adjusted_Fare'] *
    df['demand_surge_multiplier'] *
    (1 + df['distance_adjustment']) *
    df['elasticity_fare_adjustment']
)

# Define target variable and features
TARGET = 'Historical_Cost_Adjusted_Fare'
X = df[['Engineered_Fare', 'Loyalty_Discount_Applied', 'Cumulative_Past_Rides_Discount',
        'Traffic_Condition_Adjusted_Fare', 'Peak_Hours_Adjusted_Fare', 'demand_surge_multiplier',
        'distance_adjustment', 'elasticity_fare_adjustment']]
y = df[TARGET]

# Data Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)

# Calculate errors
train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)
val_rmse = mean_squared_error(y_val, y_val_pred, squared=False)
test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)

# Save results
df['Predicted_Cost'] = model.predict(scaler.transform(X))
df['Absolute_Error'] = abs(df['Historical_Cost_Adjusted_Fare'] - df['Predicted_Cost'])
df['Percentage_Error'] = (df['Absolute_Error'] / df['Historical_Cost_Adjusted_Fare']) * 100

output_files_dir = os.path.join(RESULTS_DIR, 'output_files')
output_file_path = os.path.join(output_files_dir, 'model_6_result.csv')
df.to_csv(output_file_path, index=False)
os.makedirs(output_files_dir, exist_ok=True)

result_files = r'D:\Rishi\Data\Results\result_files'
comparison_file_path = os.path.join(result_files, "comparison.csv")

# Prepare new model's metrics for Model 6
df_metrics = pd.DataFrame({
    'Models': ['Model 6'],
    'Model Description': ['Feature Engineering with Demand Surge, Distance Adjustments, Elasticity, and Loyalty Discounts'],
    'Train error': [train_rmse],
    'Val error': [val_rmse],
    'Test error': [test_rmse]
})

# Check if comparison.csv exists
if os.path.exists(comparison_file_path):
    # Read the existing comparison file
    df_combined_metrics = pd.read_csv(comparison_file_path)
    # Append the new model metrics
    df_combined_metrics = pd.concat([df_combined_metrics, df_metrics], ignore_index=True)
else:
    # If file does not exist, initialize with Model 6's metrics
    df_combined_metrics = df_metrics

# Save the updated comparison.csv file
df_combined_metrics.to_csv(comparison_file_path, index=False)