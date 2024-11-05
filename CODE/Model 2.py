import numpy as np
import pandas as pd
from Utils.constants import RAW_DIR, PROCESSED_DIR
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import root_mean_squared_error



# Load the dataset
input_file = os.path.join(RAW_DIR, "dynamic_pricing.csv")
data = pd.read_csv(input_file)


# Select features and target variable
X = data[['Number_of_Riders', 'Number_of_Drivers', 'Number_of_Past_Rides', 'Average_Ratings', 'Expected_Ride_Duration']]
y = data['Historical_Cost_of_Ride']

# Split the data into train, validation, and test sets
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
    "Lasso": Lasso(),
    "Ridge": Ridge(),
    "SVR": SVR()
}

# Dictionary to store errors
results = {
    "Model": [],
    "Train Error": [],
    "Validation Error": [],
    "Test Error": []
}

# Train each model and calculate errors
for name, model in models.items():
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
    results["Model"].append(name)
    results["Train Error"].append(train_rmse)
    results["Validation Error"].append(val_rmse)
    results["Test Error"].append(test_rmse)

# Convert results to a DataFrame
results_data = pd.DataFrame(results)

# Save the results to a CSV file
output_file = os.path.join(PROCESSED_DIR, "model2_errors.csv")
data.to_csv(output_file, index=False)

print(f"Processed data saved to {output_file})
