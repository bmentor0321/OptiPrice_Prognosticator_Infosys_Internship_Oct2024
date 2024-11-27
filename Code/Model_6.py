import os
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from Utils.constants import RAW_DIR ,RESULT_FILES_DIR, RAW_FILE_PATH, OUTPUT_FILES_DIR, LOYALTY_DISCOUNTS, BASE_FARE,PEAK_HOURS_MULTIPLIER, TRAFFIC_MULTIPLIER, PAST_RIDES_DISCOUNT, DURATION_MULTIPLIER, DURATION_THRESHOLD, HISTORICAL_COST_ADJUSTMENT_FACTOR, DISTANCE_THRESHOLD

# Constants for Directories
COMPARISON_FILE_PATH = os.path.join(RESULT_FILES_DIR, "comparison.csv")
OUTPUT_FILE_PATH = os.path.join(OUTPUT_FILES_DIR, "model_6_result.csv")
df = pd.read_csv(RAW_FILE_PATH)

# Feature Engineering
df['riders_to_drivers_ratio'] = df['Number_of_Riders'] / (df['Number_of_Drivers'] + 1)
df['duration_per_rider'] = df['Expected_Ride_Duration'] / df['Number_of_Riders']
df['log_historical_cost'] = np.log1p(df['Historical_Cost_of_Ride'])
df['rating_ride_interaction'] = df['Average_Ratings'] * df['Expected_Ride_Duration']

# Categorical encoding
df['booking_category'] = pd.Categorical(df['Time_of_Booking']).codes
df['location_code'] = pd.Categorical(df['Location_Category']).codes
df['loyalty_code'] = pd.Categorical(df['Customer_Loyalty_Status']).codes
df['vehicle_code'] = pd.Categorical(df['Vehicle_Type']).codes

# Discounts and adjustments
df['Loyalty_Discount_Applied'] = df['Customer_Loyalty_Status'].map(LOYALTY_DISCOUNTS)
df['Cumulative_Past_Rides_Discount'] = (df['Number_of_Past_Rides'] // 50) * PAST_RIDES_DISCOUNT
df['Traffic_Condition_Adjusted_Fare'] = df['Location_Category'].apply(lambda x: TRAFFIC_MULTIPLIER if x == "Heavy" else 1)
df['Peak_Hours_Adjusted_Fare'] = df['Time_of_Booking'].apply(lambda x: PEAK_HOURS_MULTIPLIER if x in ["8-10 AM", "5-8 PM"] else 1)
df['demand_surge_multiplier'] = df['riders_to_drivers_ratio'].apply(lambda x: 1.5 if x > 1 else 0.8 if x < 0.5 else 1)
df['distance_adjustment'] = (df['Expected_Ride_Duration'] - DISTANCE_THRESHOLD).apply(lambda x: 0.02 * max(0, x))
df['elasticity_fare_adjustment'] = 1 + (DURATION_MULTIPLIER * (df['Expected_Ride_Duration'] > DURATION_THRESHOLD))
df['Historical_Cost_Adjusted_Fare'] = df['Historical_Cost_of_Ride'] * HISTORICAL_COST_ADJUSTMENT_FACTOR

# Define Features and Target
dataset_columns = [ 'Number_of_Riders', 'Number_of_Drivers', 'Location_Category', 'Customer_Loyalty_Status',
    'Number_of_Past_Rides', 'Average_Ratings', 'Time_of_Booking', 'Vehicle_Type', 
    'Expected_Ride_Duration', 'Historical_Cost_of_Ride']
features = [
    'riders_to_drivers_ratio', 'duration_per_rider', 'log_historical_cost',
    'rating_ride_interaction', 'booking_category', 'location_code', 'loyalty_code', 'vehicle_code',
    'Loyalty_Discount_Applied', 'Cumulative_Past_Rides_Discount', 'Traffic_Condition_Adjusted_Fare',
    'Peak_Hours_Adjusted_Fare', 'demand_surge_multiplier', 'distance_adjustment', 
    'elasticity_fare_adjustment', 'Loyalty_Discount_Applied'
]
target = 'Historical_Cost_Adjusted_Fare'

X = df[features]
y = df[target]

# Standardize Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Create DMatrix for XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_test, label=y_test)

# Optimized XGBoost Parameters
params = {
    "objective": "reg:squarederror",
    "learning_rate": 0.03,
    "max_depth": 6,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "reg_alpha": 1.5,
    "reg_lambda": 15,
    "seed": 42
}

# Train Model with Early Stopping
evals = [(dtrain, 'train'), (dval, 'eval')]
xgb_model = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=500,
    evals=evals,
    early_stopping_rounds=30,
    verbose_eval=True
)

# Predictions
y_train_pred = xgb_model.predict(dtrain)
y_val_pred = xgb_model.predict(dval)
y_test_pred = xgb_model.predict(dtest)

# RMSE Calculation
train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)
val_rmse = mean_squared_error(y_val, y_val_pred, squared=False)
test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)

# Save Results
df['Predicted Value'] = xgb_model.predict(xgb.DMatrix(scaler.transform(X)))
df['True Value'] = y
df['Absolute_Error'] = abs(df['True Value'] - df['Predicted Value'])
df['Percentage_Error'] = (df['Absolute_Error'] / df['True Value']) * 100

output_columns = dataset_columns + features + ['True Value', 'Predicted Value', 'Absolute_Error', 'Percentage_Error']
output_df = df[output_columns]
output_df.to_csv(OUTPUT_FILE_PATH, index=False)
print(f"Model results saved to: {OUTPUT_FILE_PATH}")

# Save Metrics
df_metrics = pd.DataFrame({
    'Models': ['Model 6'],
    'Model Description': ['Integrated model using XGBoost with feature engineering, correlation-based selection, and scaling for accurate fare prediction'],
    'Train error': [train_rmse],
    'Val error': [val_rmse],
    'Test error': [test_rmse]

})

if os.path.exists(COMPARISON_FILE_PATH):
    df_existing_metrics = pd.read_csv(COMPARISON_FILE_PATH)
    df_combined_metrics = pd.concat([df_existing_metrics, df_metrics], ignore_index=True)
else:
    df_combined_metrics = df_metrics

df_combined_metrics.to_csv(COMPARISON_FILE_PATH, index=False)
print(f"Comparison metrics saved to: {COMPARISON_FILE_PATH}")

# Print Metrics
print(f"Train RMSE: {train_rmse:.2f}")
print(f"Validation RMSE: {val_rmse:.2f}")
print(f"Test RMSE: {test_rmse:.2f}")