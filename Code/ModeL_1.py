import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

data = pd.read_csv('dynamic_pricing.csv')

# Defining the feature set (X) and target variable (Y)
X = data[['Number_of_Riders', 'Number_of_Drivers', 'Number_of_Past_Rides', 'Average_Ratings', 'Expected_Ride_Duration']]
Y = data['Historical_Cost_of_Ride']


np.random.seed(42)
num_samples = 1000
num_features = 1000

# Split data into training, validation, and testing sets
X_temp, X_test, y_temp, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

# Initialize and configure models
models = {
    "Linear Regression": LinearRegression(),
    "Lasso Regression": Lasso(alpha=0.1),
    "Ridge Regression": Ridge(alpha=0.1),
    "SVR": SVR(kernel='linear')
}

# Training and validation for each model
validation_results = {}
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_val_pred = model.predict(X_val)
    validation_mse = mean_squared_error(y_val, y_val_pred)
    validation_results[model_name] = validation_mse
    print(f"{model_name} Validation MSE: {validation_mse:.2f}")

# Testing models and generating predictions
test_results = {}
for model_name, model in models.items():
    y_test_pred = model.predict(X_test)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_results[model_name] = {
        "Predictions": y_test_pred,
        "MSE": test_mse
    }

# Prepare results for saving
results_df = pd.DataFrame({
    'Actual': y_test
})

# Append predictions and errors 
for model_name, result in test_results.items():
    results_df[f'Predicted_{model_name}'] = result['Predictions']
    results_df[f'Error_{model_name}'] = results_df['Actual'] - result['Predictions']

# Save results to CSV
results_df.to_csv('Model_1_Errors.csv', index=False)

# Display results DataFrame
print(results_df.head())
