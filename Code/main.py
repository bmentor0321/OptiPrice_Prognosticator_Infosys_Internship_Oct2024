import pandas as pd
import numpy as np
import os
from Utils.constants import RAW_DIR, PROCESSED_DIR, RESULTS_DIR
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from Utils.utils import calculate_metrics, adjust_price

# Load the raw data file
input_file = os.path.join(RAW_DIR, "dynamic_pricing.csv")
df = pd.read_csv(input_file)

# Calculate metrics using imported function
df = calculate_metrics(df)

# Calculate adjusted prices
df['adjusted_price'] = df.apply(adjust_price, axis=1)

# Define categorical columns for one-hot encoding
categorical_columns = ['Location_Category', 'Time_of_Booking', 'Vehicle_Type']
numerical_columns = ['demand_metric', 'supply_metric', 'Expected_Ride_Duration']
y = df['Historical_Cost_of_Ride']

# Initialize results storage
results_data = {
    'Model': [],
    'Train_MSE': [],
    'Val_MSE': [],
    'Test_MSE': [],
    'Train_MAPE': [],
    'Val_MAPE': [],
    'Test_MAPE': []
}

### Model 1: Linear Regression with only numerical columns (no scaling, no encoding)
X_num = df[numerical_columns]
X_train, X_temp, y_train, y_temp = train_test_split(X_num, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

model1 = LinearRegression()
model1.fit(X_train, y_train)

# Predictions and metrics for Model 1
y_pred_train_1 = model1.predict(X_train)
y_pred_val_1 = model1.predict(X_val)
y_pred_test_1 = model1.predict(X_test)

# Calculate metrics for Model 1
train_mse_1 = mean_squared_error(y_train, y_pred_train_1)
val_mse_1 = mean_squared_error(y_val, y_pred_val_1)
test_mse_1 = mean_squared_error(y_test, y_pred_test_1)
train_mape_1 = mean_absolute_percentage_error(y_train, y_pred_train_1)
val_mape_1 = mean_absolute_percentage_error(y_val, y_pred_val_1)
test_mape_1 = mean_absolute_percentage_error(y_test, y_pred_test_1)

# Save Model 1 predictions
df['predicted_price_model1'] = model1.predict(X_num)
df.to_csv(os.path.join(PROCESSED_DIR, "dynamic_pricing_model1.csv"), index=False)

# Store Model 1 results
results_data['Model'].append('Linear Regression (Numerical Only)')
results_data['Train_MSE'].append(train_mse_1)
results_data['Val_MSE'].append(val_mse_1)
results_data['Test_MSE'].append(test_mse_1)
results_data['Train_MAPE'].append(train_mape_1)
results_data['Val_MAPE'].append(val_mape_1)
results_data['Test_MAPE'].append(test_mape_1)

### Model 2: Linear Regression with numerical columns and standard scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

model2 = LinearRegression()
model2.fit(X_train_scaled, y_train)

# Predictions and metrics for Model 2
y_pred_train_2 = model2.predict(X_train_scaled)
y_pred_val_2 = model2.predict(X_val_scaled)
y_pred_test_2 = model2.predict(X_test_scaled)

# Calculate metrics for Model 2
train_mse_2 = mean_squared_error(y_train, y_pred_train_2)
val_mse_2 = mean_squared_error(y_val, y_pred_val_2)
test_mse_2 = mean_squared_error(y_test, y_pred_test_2)
train_mape_2 = mean_absolute_percentage_error(y_train, y_pred_train_2)
val_mape_2 = mean_absolute_percentage_error(y_val, y_pred_val_2)
test_mape_2 = mean_absolute_percentage_error(y_test, y_pred_test_2)

# Save Model 2 predictions
df['predicted_price_model2'] = model2.predict(scaler.transform(X_num))
df.to_csv(os.path.join(PROCESSED_DIR, "dynamic_pricing_model2.csv"), index=False)

# Store Model 2 results
results_data['Model'].append('Linear Regression (Numerical + Scaling)')
results_data['Train_MSE'].append(train_mse_2)
results_data['Val_MSE'].append(val_mse_2)
results_data['Test_MSE'].append(test_mse_2)
results_data['Train_MAPE'].append(train_mape_2)
results_data['Val_MAPE'].append(val_mape_2)
results_data['Test_MAPE'].append(test_mape_2)

### Model 3: Linear Regression with one-hot encoding and standard scaling (existing code)
column_trans = make_column_transformer(
    (OneHotEncoder(sparse_output=False), categorical_columns),
    remainder='passthrough'
)

X_transformed = column_trans.fit_transform(df[categorical_columns + numerical_columns])
X_train, X_temp, y_train, y_temp = train_test_split(X_transformed, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

model3 = LinearRegression()
model3.fit(X_train_scaled, y_train)

# Predictions and metrics for Model 3
y_pred_train_3 = model3.predict(X_train_scaled)
y_pred_val_3 = model3.predict(X_val_scaled)
y_pred_test_3 = model3.predict(X_test_scaled)

# Calculate metrics for Model 3
train_mse_3 = mean_squared_error(y_train, y_pred_train_3)
val_mse_3 = mean_squared_error(y_val, y_pred_val_3)
test_mse_3 = mean_squared_error(y_test, y_pred_test_3)
train_mape_3 = mean_absolute_percentage_error(y_train, y_pred_train_3)
val_mape_3 = mean_absolute_percentage_error(y_val, y_pred_val_3)
test_mape_3 = mean_absolute_percentage_error(y_test, y_pred_test_3)

# Save Model 3 predictions
df['predicted_price_model3'] = model3.predict(scaler.transform(X_transformed))
df.to_csv(os.path.join(PROCESSED_DIR, "dynamic_pricing_model3.csv"), index=False)

# Store Model 3 results
results_data['Model'].append('Linear Regression (Numerical + Scaling + One-Hot Encoding)')
results_data['Train_MSE'].append(train_mse_3)
results_data['Val_MSE'].append(val_mse_3)
results_data['Test_MSE'].append(test_mse_3)
results_data['Train_MAPE'].append(train_mape_3)
results_data['Val_MAPE'].append(val_mape_3)
results_data['Test_MAPE'].append(test_mape_3)

# Save results summary
results_df = pd.DataFrame(results_data)
results_df.to_csv(os.path.join(RESULTS_DIR, "results_summary.csv"), index=False)

print("\nResults saved to: results_summary.csv")