import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_csv('dynamic_pricing.csv')

# Select features and target variable
X = data[['Number_of_Riders', 'Number_of_Drivers', 'Number_of_Past_Rides', 'Average_Ratings', 'Expected_Ride_Duration']]
Y = data['Historical_Cost_of_Ride']

# Generate synthetic dataset (for demo purposes)
np.random.seed(42)
num_samples = 1000
num_features = 1000

# Split the dataset into training, validation, and test sets
X_temp, X_test, y_temp, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)  # 0.25 of 0.8 = 0.2

# Initialize different models
linear_regressor = LinearRegression()
lasso_regressor = Lasso(alpha=0.1)
ridge_regressor = Ridge(alpha=0.1)
svm_regressor = SVR(kernel='linear')

# Dictionary of models for iteration
models = {
    "Linear Regression": linear_regressor,
    "Lasso Regression": lasso_regressor,
    "Ridge Regression": ridge_regressor,
    "Support Vector Regression": svm_regressor
}

# Train models and validate on the validation set
validation_mse_results = {}
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_val_pred = model.predict(X_val)
    validation_mse = mean_squared_error(y_val, y_val_pred)
    validation_mse_results[model_name] = validation_mse
    print(f"{model_name} Validation MSE: {validation_mse:.2f}")

# Test models and get predictions
test_mse_results = {}
for model_name, model in models.items():
    y_test_pred = model.predict(X_test)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_mse_results[model_name] = {
        "Predictions": y_test_pred,
        "MSE": test_mse
    }

# Create a DataFrame to save actual vs predicted results
results_comparison_df = pd.DataFrame({
    'Actual_Cost': y_test
})

# Add predictions and errors for each model
for model_name, result in test_mse_results.items():
    results_comparison_df[f'Predicted_{model_name}'] = result['Predictions']
    results_comparison_df[f'Error_{model_name}'] = results_comparison_df['Actual_Cost'] - result['Predictions']

# Save results to CSV
results_comparison_df.to_csv('ride_cost_predictions.csv', index=False)

# Display the first few rows of the results
print(results_comparison_df.head())
