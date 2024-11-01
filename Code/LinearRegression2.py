import os
from Utils.constants import RAW_DIR, RESULTS_DIR
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import joblib

# Load the dataset
file_path = os.path.join(RAW_DIR, "dynamic_pricing.csv")
df = pd.read_csv(file_path)
data=df
location_factors = {'urban': 1.1, 'suburban': 1.0}
loyalty_discounts = {'Loyal': 0.95, 'Non-loyal': 1.0}
peak_hour_factors = {'peak': 1.1, 'off-peak': 1.0}

# Dynamic price adjustment function
def dynamic_price_adjustment(historical_cost, demand, supply, location, loyalty, peak_hour):
    # Get the respective factors for location, loyalty, and peak hour
    location_factor = location_factors.get(location, 1.0)
    loyalty_discount = loyalty_discounts.get(loyalty, 1.0)
    peak_hour_factor = peak_hour_factors.get(peak_hour, 1.0)
    
    # Calculate demand-supply adjustment
    demand_supply_adjustment = 1 + 0.2 * ((demand / supply > 1.2) - (demand / supply < 0.8))
    
    # Calculate adjusted price
    adjusted_price = historical_cost * demand_supply_adjustment * location_factor * loyalty_discount * peak_hour_factor
    return round(adjusted_price, 2)

# Apply the function row-wise to create a new column
data['Adjusted_Cost'] = data.apply(lambda row: dynamic_price_adjustment(
    row['Historical_Cost_of_Ride'],
    row['Number_of_Riders'],
    row['Number_of_Drivers'],
    row['Location_Category'],
    row['Customer_Loyalty_Status'],
    row['Time_of_Booking']
), axis=1)

# Display the updated DataFrame
data


# Apply Label Encoding to all categorical columns
categorical_columns = ['Location_Category', 'Customer_Loyalty_Status', 'Time_of_Booking', 'Vehicle_Type']
label_encoders = {}  # Dictionary to store encoders for each column

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Save the encoder to decode later if needed

# Separate features (X) and target (y)
X = df.drop(columns=['Historical_Cost_of_Ride', 'Adjusted_Cost'])
y = df['Adjusted_Cost']

# Normalize the independent variables
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

# Split the data
X_train, X_temp, y_train, y_temp = train_test_split(X_normalized, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Validate the model
y_val_pred = model.predict(X_val)
val_mae = mean_absolute_error(y_val, y_val_pred)
val_mse = mean_squared_error(y_val, y_val_pred)
print(f"Validation MAE: {val_mae}, Validation MSE: {val_mse}")

# Test the model
y_test_pred = model.predict(X_test)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
print(f"Test MAE: {test_mae}, Test MSE: {test_mse}")

# Calculate the error
errors = y_test - y_test_pred

# Save Test dataset + Y variable + Y(predicted) + Error into a CSV file
test_results = pd.DataFrame(X_test, columns=X.columns)
test_results['Y_True'] = y_test
test_results['Y_Predicted'] = y_test_pred
test_results['Error'] = errors

output_file = os.path.join(RESULTS_DIR, "Linear_Regression_result2.csv")
test_results.to_csv(output_file, index=False)

print(f"Output data saved to {output_file}")
