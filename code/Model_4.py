import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import numpy as np

# Define constants
DATASET_PATH = "C:/Users/chide/OneDrive/Desktop/infosys/Dataset/dynamic_pricing.csv"
FEATURES_TO_SCALE = ['Number_of_Riders', 'Number_of_Drivers', 'Expected_Ride_Duration']
TARGET_VARIABLE = "Historical_Cost_of_Ride"
ADJUSTMENT_FACTOR = 1.1  # Example adjustment factor

# Feature Engineering Function
def feature_engineering(X):
    # Polynomial features for selected variables
    poly = PolynomialFeatures(degree=2, include_bias=False)
    poly_features = poly.fit_transform(X[['Number_of_Riders', 'Number_of_Drivers', 'Expected_Ride_Duration']])
    
    # Manually create feature names if get_feature_names_out is not available
    feature_names = poly.get_feature_names(['Number_of_Riders', 'Number_of_Drivers', 'Expected_Ride_Duration'])
    poly_df = pd.DataFrame(poly_features, columns=feature_names, index=X.index)
    
    # Interaction terms
    X['Riders_x_Drivers'] = X['Number_of_Riders'] * X['Number_of_Drivers']
    X['Riders_x_Duration'] = X['Number_of_Riders'] * X['Expected_Ride_Duration']
    X['Drivers_x_Duration'] = X['Number_of_Drivers'] * X['Expected_Ride_Duration']
    
    # Log transformation to reduce skewness in variables
    X['Log_Number_of_Riders'] = np.log1p(X['Number_of_Riders'])
    X['Log_Number_of_Drivers'] = np.log1p(X['Number_of_Drivers'])
    X['Log_Expected_Ride_Duration'] = np.log1p(X['Expected_Ride_Duration'])
    
    # Combine original features with engineered features
    X = pd.concat([X, poly_df], axis=1)
    
    return X

# Data Preprocessing
def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)

    # Apply adjustment to the target variable
    df["Adjusted_Cost_of_Ride"] = df[TARGET_VARIABLE] * ADJUSTMENT_FACTOR

    # Selecting numerical features and scaling
    X = df.select_dtypes(include=['float64', 'int64']).drop(columns=[TARGET_VARIABLE, "Adjusted_Cost_of_Ride"])
    y = df["Adjusted_Cost_of_Ride"]
    
    # Feature engineering
    X = feature_engineering(X)

    # Feature scaling
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
    
    try:
        results.to_csv(save_path, index=False)
        print(f"Results saved to {save_path}")
    except Exception as e:
        print(f"Error saving file: {e}")
    
    return val_error, test_error

# Main script
if __name__ == "__main__":
    file_path = DATASET_PATH
    save_path = "C:/Users/chide/OneDrive/Desktop/infosys/Model_4_test_result_with_error.csv"
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_preprocess_data(file_path)
    
    if X_train is not None:
        val_error, test_error = train_validate_test_model(X_train, X_val, X_test, y_train, y_val, y_test, save_path)
        print(f'Validation RMSE: {val_error}')
        print(f'Test RMSE: {test_error}')
