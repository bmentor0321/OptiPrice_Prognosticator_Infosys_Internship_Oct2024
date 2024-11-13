import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

data = pd.read_csv('dynamic_pricing.csv')

X = data[['Number_of_Riders', 'Number_of_Drivers', 'Number_of_Past_Rides', 'Average_Ratings', 'Expected_Ride_Duration']]
Y = data['Historical_Cost_of_Ride']

# Generate synthetic dataset
np.random.seed(42)
num_samples = 1000
num_features = 1000


# Split the dataset into train, validation, and test sets
X_temp, X_test, y_temp, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)  # 0.25 of 0.8 = 0.2

# Initialize models
linear_model = LinearRegression()
lasso_model = Lasso(alpha=0.1)   # A linear regression model with L1 regularization (helps prevent overfitting).
ridge_model = Ridge(alpha=0.1)   # A linear regression model with L2 regularization (also helps prevent overfitting).
svr_model = SVR(kernel='linear')  #  A support vector regression model with a linear kernel.

# Train models
models = {
    "Linear Regression": linear_model,
    "Lasso Regression": lasso_model,
    "Ridge Regression": ridge_model,
    "SVR": svr_model
}

# Fit models and validate
validation_results = {}
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_val_pred = model.predict(X_val)
    validation_mse = mean_squared_error(y_val, y_val_pred)
    validation_results[model_name] = validation_mse
    print(f"{model_name} Validation MSE: {validation_mse:.2f}")

# Test models and get predictions
test_results = {}
for model_name, model in models.items():
    y_test_pred = model.predict(X_test)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_results[model_name] = {
        "Predictions": y_test_pred,
        "MSE": test_mse
    }

# Create a DataFrame to save results
results_df = pd.DataFrame({
    'Actual': y_test
})

# Add predictions for each model
for model_name, result in test_results.items():
    results_df[f'Predicted_{model_name}'] = result['Predictions']
    results_df[f'Error_{model_name}'] = results_df['Actual'] - result['Predictions']

# Save to CSV
results_df.to_csv('ride_cost_predictions.csv', index=False)

# Display the first few rows of the results
print(results_df.head())
