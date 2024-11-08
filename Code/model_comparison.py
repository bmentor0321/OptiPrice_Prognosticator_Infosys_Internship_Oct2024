# Import necessary libraries
import pandas as pd
import numpy as np
import os
from Utils.constants import RAW_DIR, PROCESSED_DIR
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.metrics import mean_absolute_percentage_error

# Load the raw data file
input_file = os.path.join(RAW_DIR, "dynamic_pricing.csv")
df = pd.read_csv(input_file)
# Load the raw data
input_file = os.path.join(RAW_DIR, "dynamic_pricing.csv")
df = pd.read_csv(input_file)

# Feature Engineering: Calculate demand-supply ratio, demand and supply classifications, and metrics
df["d_s_ratio"] = round(df["Number_of_Riders"] / df["Number_of_Drivers"], 2)
df["demand_class"] = np.where(df["Number_of_Riders"] > np.percentile(df["Number_of_Riders"], 75), "high_demand", "low_demand")
df["supply_class"] = np.where(df["Number_of_Drivers"] > np.percentile(df["Number_of_Drivers"], 75), "high_supply", "low_supply")

df["demand_metric"] = np.where(df["demand_class"] == "high_demand",
                               df["Number_of_Riders"] / np.percentile(df["Number_of_Riders"], 75),
                               df["Number_of_Riders"] / np.percentile(df["Number_of_Riders"], 25))
df["supply_metric"] = np.where(df["supply_class"] == "high_supply",
                               df["Number_of_Drivers"] / np.percentile(df["Number_of_Drivers"], 75),
                               df["Number_of_Drivers"] / np.percentile(df["Number_of_Drivers"], 25))

# Define features and target variable
X = df[['Number_of_Riders', 'Number_of_Drivers', 'Expected_Ride_Duration', 'demand_metric', 'supply_metric', 
        'Location_Category', 'Time_of_Booking', 'Vehicle_Type']]
y = df['Historical_Cost_of_Ride']

# Identify categorical columns
categorical_columns = ['Location_Category', 'Time_of_Booking', 'Vehicle_Type']

# Initialize an empty list to store results
results = []

# Define a function to evaluate and store model performance using MAPE
def evaluate_model(X_train, X_val, X_test, y_train, y_val, y_test, model_desc):
    model = LinearRegression()
    model.fit(X_train, y_train)

    train_mape = mean_absolute_percentage_error(y_train, model.predict(X_train))
    val_mape = mean_absolute_percentage_error(y_val, model.predict(X_val))
    test_mape = mean_absolute_percentage_error(y_test, model.predict(X_test))

    results.append({
        'Model_Description': model_desc,
        'Train_MAPE': train_mape,
        'Val_MAPE': val_mape,
        'Test_MAPE': test_mape
    })

# Split data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# 1. Linear Regression with Numerical Features Only
evaluate_model(X_train[['Number_of_Riders', 'Number_of_Drivers', 'Expected_Ride_Duration', 'demand_metric', 'supply_metric']],
               X_val[['Number_of_Riders', 'Number_of_Drivers', 'Expected_Ride_Duration', 'demand_metric', 'supply_metric']],
               X_test[['Number_of_Riders', 'Number_of_Drivers', 'Expected_Ride_Duration', 'demand_metric', 'supply_metric']],
               y_train, y_val, y_test, "LR with Numerical Features Only")

# 2. Linear Regression with One-Hot Encoding
column_trans = make_column_transformer((OneHotEncoder(sparse_output=False), categorical_columns), remainder='passthrough')
X_transformed = column_trans.fit_transform(X)
X_train, X_temp, y_train, y_temp = train_test_split(X_transformed, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
evaluate_model(X_train, X_val, X_test, y_train, y_val, y_test, "LR with One-Hot Encoding")

# 3. Linear Regression with Standard Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
evaluate_model(X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, "LR with Standard Scaling")

# 4. Linear Regression with Scaling and One-Hot Encoding
X_scaled_encoded = column_trans.fit_transform(X)
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled_encoded, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
evaluate_model(X_train, X_val, X_test, y_train, y_val, y_test, "LR with Scaling and One-Hot Encoding")

# Save results to a CSV file
results_df = pd.DataFrame(results)
results_file = os.path.join(PROCESSED_DIR, "model_comparison_results.csv")
results_df.to_csv(results_file, index=False)


print(f"Model comparison results saved to: {results_file}")
