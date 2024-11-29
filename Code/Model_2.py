import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

data = pd.read_csv("dynamic_pricing.csv")

# Features and target variable
X = data[['Number_of_Riders', 'Number_of_Drivers', 'Number_of_Past_Rides', 'Average_Ratings', 'Expected_Ride_Duration']]
y = data['Historical_Cost_of_Ride']

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Initialize models
models = {
    "Linear Regression": LinearRegression(),
    "Lasso Regression": Lasso(),
    "Ridge Regression": Ridge(),
    "Support Vector Regression": SVR()
}

# Dictionary to store errors
results = {
    "Model": [],
    "Training RMSE": [],
    "Validation RMSE": [],
    "Testing RMSE": []
}

# Train each model and calculate errors
for model_name, model in models.items():
    # Fit the model on the training data
    model.fit(X_train, y_train)

    # Predict on train, validation, and test sets
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    # Calculate RMSE for train, validation, and test sets
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    # Store the results
    results["Model"].append(model_name)
    results["Training RMSE"].append(train_rmse)
    results["Validation RMSE"].append(val_rmse)
    results["Testing RMSE"].append(test_rmse)

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Save the results to a CSV file
results_df.to_csv('Model_2_Errors.csv', index=False)

print("Processed data saved as 'model_2_errors.csv'")
