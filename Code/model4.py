import pandas as pd
import numpy as np
import os
from Utils.constants import RAW_DIR, PROCESSED_DIR, RESULTS_DIR
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from Utils.utils import calculate_metrics

# Load and preprocess data
input_file = os.path.join(RAW_DIR, "dynamic_pricing.csv")
df = pd.read_csv(input_file)
df = calculate_metrics(df)

# Prepare features
categorical_columns = ['Location_Category', 'Time_of_Booking', 'Vehicle_Type']
numerical_columns = ['demand_metric', 'supply_metric', 'Expected_Ride_Duration']
y = df['Historical_Cost_of_Ride']

# Create preprocessing pipeline
transformer = ColumnTransformer(transformers=[
    ('categorical', OneHotEncoder(sparse_output=False), categorical_columns),
    ('numerical', StandardScaler(), numerical_columns)
])

# Create full pipeline with PCA
pipeline = Pipeline(steps=[
    ('preprocessing', transformer),
    ('pca', PCA(n_components=0.95))
])

# Prepare data splits
X = df[categorical_columns + numerical_columns]
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Fit and transform data with pipeline
X_train_pca = pipeline.fit_transform(X_train)
X_val_pca = pipeline.transform(X_val)
X_test_pca = pipeline.transform(X_test)

# Train linear regression model
model = LinearRegression()
model.fit(X_train_pca, y_train)

# Make predictions
y_pred_train = model.predict(X_train_pca)
y_pred_val = model.predict(X_val_pca)
y_pred_test = model.predict(X_test_pca)

# Calculate RMSE metrics
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

train_mape = mean_absolute_percentage_error(y_train, y_pred_train)
val_mape = mean_absolute_percentage_error(y_val, y_pred_val)
test_mape = mean_absolute_percentage_error(y_test, y_pred_test)

# Save predictions
X_all_pca = pipeline.transform(X)
df['predicted_price_pca'] = model.predict(X_all_pca)
df.to_csv(os.path.join(PROCESSED_DIR, "dynamic_pricing_model4.csv"), index=False)

# Read existing results summary
results_file = os.path.join(RESULTS_DIR, "results_summary.csv")
results_df = pd.read_csv(results_file)

# Add new model results
new_result = pd.DataFrame({
    'Model_Name': ['LR+OneHot+SD+PCA'],
    'Model_Summary': ['Linear regression with PCA transformation'],
    'Train_Error': [train_rmse],
    'Val_Error': [val_rmse],
    'Test_Error': [test_rmse]
})

# Update results summary
results_df = pd.concat([results_df, new_result], ignore_index=True)
results_df.to_csv(results_file, index=False)

print("\nResults have been added to results_summary.csv")
print("\nModel 4 (PCA) Performance Metrics:")
print(f"Training RMSE: {train_rmse:.2f}")
print(f"Validation RMSE: {val_rmse:.2f}")
print(f"Testing RMSE: {test_rmse:.2f}")
print(f"Training MAPE: {train_mape:.2%}")
print(f"Validation MAPE: {val_mape:.2%}")
print(f"Testing MAPE: {test_mape:.2%}")

# Print number of components selected by PCA
n_components = pipeline.named_steps['pca'].n_components_
print(f"\nNumber of PCA components selected (95% variance): {n_components}")