import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Define constants
DATASET_PATH = "C:/Users/chide/OneDrive/Desktop/infosys/Dataset/dynamic_pricing.csv"
FEATURES_TO_SCALE = ['Number_of_Riders', 'Number_of_Drivers', 'Expected_Ride_Duration']
TARGET_VARIABLE = "Historical_Cost_of_Ride"
ADJUSTMENT_FACTOR = 1.1  # Example adjustment factor

# Data Preprocessing
def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)

    # Apply adjustment to the target variable
    df["Adjusted_Cost_of_Ride"] = df[TARGET_VARIABLE] * ADJUSTMENT_FACTOR

    # Selecting numerical features and scaling
    X = df.select_dtypes(include=['float64', 'int64']).drop(columns=[TARGET_VARIABLE, "Adjusted_Cost_of_Ride"])
    y = df["Adjusted_Cost_of_Ride"]

    scaler = StandardScaler()
    X[FEATURES_TO_SCALE] = scaler.fit_transform(X[FEATURES_TO_SCALE])

    # Splitting the dataset
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test

# Model Training, Validation, Testing
def train_validate_test_model(X_train, X_val, X_test, y_train, y_val, y_test, save_path):
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Validation
    y_pred_val = model.predict(X_val)
    val_error = mean_squared_error(y_val, y_pred_val, squared=False)

    # Testing
    y_pred_test = model.predict(X_test)
    test_error = mean_squared_error(y_test, y_pred_test, squared=False)

    # Save results
    results = X_test.copy()
    results["Actual_Adjusted_Cost"] = y_test
    results["Predicted_Adjusted_Cost"] = y_pred_test
    results["Error"] = results["Actual_Adjusted_Cost"] - results["Predicted_Adjusted_Cost"]
    results.to_csv(save_path, index=False)
    
    return val_error, test_error

# Main script
if __name__ == "__main__":
    file_path = DATASET_PATH
    save_path = "C:/Users/chide/OneDrive/Desktop/infosys/Model_4_test_result_with_error.csv"
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_preprocess_data(file_path)
    val_error, test_error = train_validate_test_model(X_train, X_val, X_test, y_train, y_val, y_test, save_path)
    print(f'Validation RMSE: {val_error}')
    print(f'Test RMSE: {test_error}')
