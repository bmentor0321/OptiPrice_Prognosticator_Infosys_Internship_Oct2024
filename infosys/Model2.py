# Update the content of the new file with the user's specified path and variables
# Assuming similar structure as Model_1.py

# Define the updated content for Model_2.py
updated_model_2_content = """
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Define constants with updated path, features to scale, and target variable
DATASET_PATH = 'C:/Users/chide/OneDrive/Desktop/infosys/Dataset/dynamic_pricing.csv'
FEATURES_TO_SCALE = ['Number_of_Riders', 'Number_of_Drivers', 'Expected_Ride_Duration']
TARGET_VARIABLE = 'Historical_Cost_of_Ride'

# Data Preprocessing
def load_and_preprocess_data(filepath):
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: The file at {filepath} was not found.")
        return None, None, None, None, None, None

    # Selecting numerical features and scaling specified features
    X = df.select_dtypes(include=['float64', 'int64']).drop(columns=[TARGET_VARIABLE])
    y = df[TARGET_VARIABLE]
    
    # Feature scaling
    scaler = StandardScaler()
    X[FEATURES_TO_SCALE] = scaler.fit_transform(X[FEATURES_TO_SCALE])

    # Splitting the dataset
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test

# Model Training
def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# Model Validation
def validate_model(model, X_val, y_val):
    y_pred_val = model.predict(X_val)
    val_error = mean_squared_error(y_val, y_pred_val, squared=False)  # RMSE calculation
    return val_error

# Model Testing
def test_model(model, X_test, y_test, save_path):
    y_pred_test = model.predict(X_test)
    test_error = mean_squared_error(y_test, y_pred_test, squared=False)  # RMSE calculation

    results = X_test.copy()
    results[TARGET_VARIABLE] = y_test
    results['Predicted_Cost_of_Ride'] = y_pred_test
    results['Error'] = results[TARGET_VARIABLE] - results['Predicted_Cost_of_Ride']

    # Save results to CSV
    try:
        results.to_csv(save_path, index=False)
        print(f"Results saved to {save_path}")
    except Exception as e:
        print(f"Error saving file: {e}")

    return test_error

# Calculate training error
def calculate_training_error(model, X_train, y_train):
    y_pred_train = model.predict(X_train)
    train_error = mean_squared_error(y_train, y_pred_train, squared=False)  # RMSE calculation
    return train_error

# Main Script
if __name__ == "__main__":
    # Load and preprocess data
    file_path = DATASET_PATH
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_preprocess_data(file_path)
    
    if X_train is not None:
        # Train the model
        model = train_model(X_train, y_train)

        # Calculate training error
        training_error = calculate_training_error(model, X_train, y_train)
        print(f'Training RMSE: {training_error}')

        # Validate the model
        validation_error = validate_model(model, X_val, y_val)
        print(f'Validation RMSE: {validation_error}')

        # Test the model and save results
        save_path = os.path.join("C:", "Users", "chide", "OneDrive", "Desktop", "infosys", "Model_2_test_result_with_error.csv")
        test_error = test_model(model, X_test, y_test, save_path)
        print(f'Test RMSE: {test_error}')
"""

# Save the updated Model_2 content into a new file
updated_model_2_path = "/mnt/data/Updated_Model_2.py"
with open(updated_model_2_path, "w") as file:
    file.write(updated_model_2_content)

updated_model_2_path
