import pandas as pd
import numpy as np
import os
from Utils.constants import RAW_DIR, PROCESSED_DIR, RESULTS_DIR
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from Utils.utils import calculate_metrics, calculate_feature_engineering, calculate_dynamic_price

# Load data
input_file = os.path.join(RAW_DIR, "dynamic_pricing.csv")
df = pd.read_csv(input_file)

# Calculate all metrics and features
df = calculate_metrics(df)
df = calculate_feature_engineering(df)
df = calculate_dynamic_price(df)

# Prepare features
categorical_columns = ['Location_Category', 'Time_of_Booking', 'Vehicle_Type']
numerical_columns = [
    'demand_metric',
    'supply_metric',
    'Expected_Ride_Duration',
    'd_s_ratio',
    'Cost_Efficiency_Vehicle',
    'Cost_Efficiency_Location',
    'Location_Demand_Efficiency',
    'Time_Efficiency'
]

# Create preprocessing pipeline
transformer = ColumnTransformer(transformers=[
    ('categorical', OneHotEncoder(sparse_output=False), categorical_columns),
    ('numerical', StandardScaler(), numerical_columns)
])

# Prepare data
X = df[categorical_columns + numerical_columns]
y = df['base_price_branch']  # Using base_price_branch as target

# Split data
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Transform features
X_train_transformed = transformer.fit_transform(X_train)
X_val_transformed = transformer.transform(X_val)
X_test_transformed = transformer.transform(X_test)

# Train model
model = LinearRegression()
model.fit(X_train_transformed, y_train)

# Make predictions
y_pred_train = model.predict(X_train_transformed)
y_pred_val = model.predict(X_val_transformed)
y_pred_test = model.predict(X_test_transformed)

# Calculate RMSE metrics
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

# Make predictions for entire dataset
df['predicted_price'] = model.predict(transformer.transform(X))

# Calculate total costs and differences
total_historical_cost = df['Historical_Cost_of_Ride'].sum()
total_base_branch = df['base_price_branch'].sum()
total_predicted_cost = df['predicted_price'].sum()

# Calculate differences
base_difference = total_base_branch - total_historical_cost
base_percentage = (base_difference / total_historical_cost) * 100

pred_difference = total_predicted_cost - total_historical_cost
pred_percentage = (pred_difference / total_historical_cost) * 100

# Print detailed analysis
print("\nCost Analysis:")
print(f"Total Historical Cost: ₹{total_historical_cost:,.2f}")
print(f"\nBase Price Branch Analysis:")
print(f"Total Base Branch Cost: ₹{total_base_branch:,.2f}")
print(f"Difference from Historical: ₹{base_difference:,.2f}")
print(f"Percentage Change: {base_percentage:,.2f}%")

print(f"\nModel Prediction Analysis:")
print(f"Total Predicted Cost: ₹{total_predicted_cost:,.2f}")
print(f"Difference from Historical: ₹{pred_difference:,.2f}")
print(f"Percentage Change: {pred_percentage:,.2f}%")

print("\nModel 5 (Feature Engineering) Performance Metrics:")
print(f"Training RMSE: ₹{train_rmse:.2f}")
print(f"Validation RMSE: ₹{val_rmse:.2f}")
print(f"Testing RMSE: ₹{test_rmse:.2f}")

# Create results directory if it doesn't exist
os.makedirs(RESULTS_DIR, exist_ok=True)

# Initialize or read results summary
results_file = os.path.join(RESULTS_DIR, "results_summary.csv")
try:
    results_df = pd.read_csv(results_file)
except FileNotFoundError:
    results_df = pd.DataFrame(columns=['Model_Name', 'Model_Summary', 'Train_Error', 'Val_Error', 'Test_Error'])

# Add new model results
new_result = pd.DataFrame({
    'Model_Name': ['LR+FE+Metrics'],
    'Model_Summary': ['Linear regression with comprehensive feature engineering'],
    'Train_Error': [train_rmse],
    'Val_Error': [val_rmse],
    'Test_Error': [test_rmse]
})

# Update results summary
results_df = pd.concat([results_df, new_result], ignore_index=True)
results_df.to_csv(results_file, index=False)

# Save predictions and actual prices for analysis
df.to_csv(os.path.join(PROCESSED_DIR, "dynamic_pricing_model5.csv"), index=False)

# Print feature importance
feature_names = (transformer.named_transformers_['categorical']
                .get_feature_names_out(categorical_columns).tolist() + numerical_columns)
coefficients = pd.DataFrame(
    {'Feature': feature_names,
     'Coefficient': model.coef_}
)
print("\nTop 10 Most Important Features:")
print(coefficients.sort_values('Coefficient', ascending=False).head(10))