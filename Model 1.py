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

# Split the data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Initialize models
models = {
    'LinearRegression': LinearRegression(),
    'Lasso': Lasso(),
    'Ridge': Ridge(),
    'SupportVectorRegressor': SVR()
}

# Initialize an empty list to store results
results = []

# Train each model and calculate errors
for model_name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)

    # Calculate training error
    y_train_pred = model.predict(X_train)
    train_error = root_mean_squared_error(y_train, y_train_pred)  # RMSE

    # Calculate validation error
    y_val_pred = model.predict(X_val)
    val_error = root_mean_squared_error(y_val, y_val_pred)  # RMSE

    # Calculate testing error
    y_test_pred = model.predict(X_test)
    test_error = root_mean_squared_error(y_test, y_test_pred)  # RMSE

    # Append results for this model
    results.append({
        'Model': model_name,
        'Train Error': train_error,
        'Validation Error ': val_error,
        'Test Error ': test_error
    })

# Convert results to a DataFrame
results_data = pd.DataFrame(results)

# Save the results to a CSV file
output_file = os.path.join(PROCESSED_DIR, "model1_errors.csv")
data.to_csv(output_file, index=False)

print(f"Processed data saved to {output_file})
