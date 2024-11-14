import os
import pandas as pd
from Utils.constants import RAW_DIR, PROCESSED_DIR
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

#Define file paths
RAW_DIR = 'Data/Raw'
PROCESSED_DIR = 'Data/Processed'
input_file = os.path.join(RAW_DIR, "dynamic_pricing.csv")


# Load the raw data file
df = pd.read_csv(input_file)

# Prepare an empty list to store results for each model
results = []

# Function to calculate RMSE
def calculate_rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, squared=False)

### Model 1: All Numerical Features
X = df[['Number_of_Riders', 'Number_of_Drivers', 'Number_of_Past_Rides', 'Average_Ratings', 'Expected_Ride_Duration']]
y = df['Historical_Cost_of_Ride']

# Split the data
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Train model
model1 = LinearRegression()
model1.fit(X_train, y_train)

# Calculate errors
train_rmse = calculate_rmse(y_train, model1.predict(X_train))
val_rmse = calculate_rmse(y_val, model1.predict(X_val))
test_rmse = calculate_rmse(y_test, model1.predict(X_test))

# Append results
results.append({
    'Model': 'Model 1',
    'Description': 'All numerical features',
    'Train RMSE': train_rmse,
    'Validation RMSE': val_rmse,
    'Test RMSE': test_rmse
})

### Model 2: Numerical Features + Standard Scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

model2 = LinearRegression()
model2.fit(X_train_scaled, y_train)

# Calculate errors
train_rmse = calculate_rmse(y_train, model2.predict(X_train_scaled))
val_rmse = calculate_rmse(y_val, model2.predict(X_val_scaled))
test_rmse = calculate_rmse(y_test, model2.predict(X_test_scaled))

# Append results
results.append({
    'Model': 'Model 2',
    'Description': 'All numerical features + Standard Scaler',
    'Train RMSE': train_rmse,
    'Validation RMSE': val_rmse,
    'Test RMSE': test_rmse
})

### Model 3: Numerical Features + Standard Scaler + One-Hot Encoded Categorical Features

categorical_features = ['Vehicle_Type'] if 'Vehicle_Type' in df.columns else []
numerical_features = X.columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ],
    remainder='passthrough'
)

X_train_processed = preprocessor.fit_transform(X_train)
X_val_processed = preprocessor.transform(X_val)
X_test_processed = preprocessor.transform(X_test)

model3 = LinearRegression()
model3.fit(X_train_processed, y_train)

# Calculate errors
train_rmse = calculate_rmse(y_train, model3.predict(X_train_processed))
val_rmse = calculate_rmse(y_val, model3.predict(X_val_processed))
test_rmse = calculate_rmse(y_test, model3.predict(X_test_processed))

# Append results
results.append({
    'Model': 'Model 3',
    'Description': 'All numerical features + Standard Scaler + Categorical Features One-Hot Encoded',
    'Train RMSE': train_rmse,
    'Validation RMSE': val_rmse,
    'Test RMSE': test_rmse
})

### Model 4: Numerical Features + Standard Scaler + One-Hot Encoded Categorical Features + Engineered Features
df['Cost_Per_Minute'] = df['Historical_Cost_of_Ride'] / df['Expected_Ride_Duration']
df['Location_Demand'] = df['Number_of_Riders'] / df['Number_of_Drivers']
if 'Vehicle_Type' in df.columns:
    df['Avg_Cost_Per_Vehicle_Type'] = df.groupby('Vehicle_Type')['Historical_Cost_of_Ride'].transform('mean')

# Updated features including engineered ones
X = df[['Number_of_Riders', 'Number_of_Drivers', 'Number_of_Past_Rides', 'Average_Ratings', 
          'Expected_Ride_Duration', 'Cost_Per_Minute', 'Location_Demand']]
if 'Vehicle_Type' in df.columns:
    X['Avg_Cost_Per_Vehicle_Type'] = df['Avg_Cost_Per_Vehicle_Type']

# Re-split the data
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Process the updated features
X_train_processed = preprocessor.fit_transform(X_train)
X_val_processed = preprocessor.transform(X_val)
X_test_processed = preprocessor.transform(X_test)

model4 = LinearRegression()
model4.fit(X_train_processed, y_train)

# Calculate errors
train_rmse = calculate_rmse(y_train, model4.predict(X_train_processed))
val_rmse = calculate_rmse(y_val, model4.predict(X_val_processed))
test_rmse = calculate_rmse(y_test, model4.predict(X_test_processed))

# Append results
results.append({
    'Model': 'Model 4',
    'Description': 'All numerical features + Standard Scaler + Categorical Features One-Hot Encoded + Engineered Features',
    'Train RMSE': train_rmse,
    'Validation RMSE': val_rmse,
    'Test RMSE': test_rmse
})

# Save results to CSV
results_df = pd.DataFrame(results)
output_file = os.path.join(PROCESSED_DIR, "model_comparison_results.csv")
results_df.to_csv(output_file, index=False)

print(f"Results saved to {output_file}")
print(results_df)
