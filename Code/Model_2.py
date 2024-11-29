import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from constants import DATASET_PATH, FEATURES_TO_SCALE, TARGET_VARIABLE

# Data Preprocessing
def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)

    # Selecting numerical features only
    X = df.select_dtypes(include=['float64', 'int64']).drop(columns=[TARGET_VARIABLE])
    y = df[TARGET_VARIABLE]

    # Splitting the dataset
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Feature scaling for selected columns
    scaler = StandardScaler()
    numerical_features = FEATURES_TO_SCALE
    X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
    X_val[numerical_features] = scaler.transform(X_val[numerical_features])
    X_test[numerical_features] = scaler.transform(X_test[numerical_features])

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
def test_model(model, X_test, y_test):
    # Make predictions
    y_pred_test = model.predict(X_test)

    # Calculate RMSE
    test_error = mean_squared_error(y_test, y_pred_test, squared=False)  # RMSE calculation

    results = X_test.copy()
    results[TARGET_VARIABLE] = y_test
    results['Predicted_Cost_of_Ride'] = y_pred_test
    results['Error'] = results[TARGET_VARIABLE] - results['Predicted_Cost_of_Ride']

    # Save results to CSV
    results.to_csv('C:/Users/viroc/Documents/Infosys Springboard Internship/Project/OptiPrice_Prognosticator_Infosys_Internship_Oct2024/Dataset/Model_2_test_results_with_error.csv', index=False)

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

    # Train the model
    model = train_model(X_train, y_train)

    # Calculate training error
    training_error = calculate_training_error(model, X_train, y_train)
    print(f'Training RMSE: {training_error}')

    # Validate the model
    validation_error = validate_model(model, X_val, y_val)
    print(f'Validation RMSE: {validation_error}')

    # Test the model and save results
    test_error = test_model(model, X_test, y_test)
    print(f'Test RMSE: {test_error}')
