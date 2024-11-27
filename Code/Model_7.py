# Import necessary libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from Utils.constants import RAW_DIR ,RESULT_FILES_DIR, RAW_FILE_PATH, OUTPUT_FILES_DIR, LOYALTY_DISCOUNTS, BASE_FARE,PEAK_HOURS_MULTIPLIER, TRAFFIC_MULTIPLIER, PAST_RIDES_DISCOUNT, DURATION_MULTIPLIER, DURATION_THRESHOLD, HISTORICAL_COST_ADJUSTMENT_FACTOR, DISTANCE_THRESHOLD

# Feature Engineering and Preprocessing
def load_and_preprocess_data(filepath):
    # Load the data
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()  # Clean column names

    # Feature Engineering: Create custom features
    TARGET_VARIABLE = 'Historical_Cost_of_Ride'
    df["Rider_Driver_Ratio"] = df["Number_of_Riders"] / (df["Number_of_Drivers"] + 1)
    df["Log_Ride_Duration"] = np.log(df["Expected_Ride_Duration"] + 1)
    df["Square_Cost"] = df[TARGET_VARIABLE] ** 2

    # Define target variable (log-transformed Adjusted Historical Cost of Ride)
    df['Adjusted_Historical_Cost_of_Ride'] = np.log1p(df[TARGET_VARIABLE])

    # Drop unnecessary columns
    df.drop(['Average_Ratings', 'Rating_Category', TARGET_VARIABLE], axis=1, inplace=True, errors='ignore')

    # One-hot encoding for categorical features
    combined_features = ['Customer_Loyalty_Status', 'Location_Category', 'Vehicle_Type', 'Time_of_Booking']
    df_encoded = pd.get_dummies(df, columns=combined_features)

    # Correlation Analysis
    correlation_matrix = df_encoded.corr()
    target_correlation = correlation_matrix['Adjusted_Historical_Cost_of_Ride'].sort_values(ascending=False)

    # Set a correlation threshold to filter features
    correlation_threshold = 0.15  # Moderate threshold
    selected_features = target_correlation[abs(target_correlation) > correlation_threshold].index.tolist()
    selected_features.remove('Adjusted_Historical_Cost_of_Ride')  # Exclude the target variable

    # Prepare features (X) and target (y)
    X = df_encoded[selected_features]
    y = df_encoded['Adjusted_Historical_Cost_of_Ride']

    # Scale numerical features
    scaler = StandardScaler()
    numerical_features = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
    X[numerical_features] = scaler.fit_transform(X[numerical_features])

    # Split data into train, validation, and test sets (60% train, 20% validation, 20% test)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test, selected_features, df

# Model Training, Validation, Testing
def train_validate_test_model(X_train, X_val, X_test, y_train, y_val, y_test, selected_features, df, save_path):
    # Gradient Boosting Regressor with Hyperparameter Tuning
    gbr = GradientBoostingRegressor(random_state=42)
    param_dist = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0],
        'min_samples_split': [2, 5, 10]
    }
    random_search = RandomizedSearchCV(gbr, param_distributions=param_dist, n_iter=50, cv=3, random_state=42, scoring='neg_mean_squared_error', n_jobs=-1)
    random_search.fit(X_train, y_train)

    # Best model from RandomizedSearchCV
    best_model = random_search.best_estimator_

    # Predictions
    y_train_pred = np.expm1(best_model.predict(X_train))  # Inverse log transformation
    y_val_pred = np.expm1(best_model.predict(X_val))
    y_test_pred = np.expm1(best_model.predict(X_test))

    y_train_actual = np.expm1(y_train)  # Actual values
    y_val_actual = np.expm1(y_val)
    y_test_actual = np.expm1(y_test)

    # Calculate RMSE for Train, Validation, and Test sets
    train_rmse = mean_squared_error(y_train_actual, y_train_pred, squared=False)
    val_rmse = mean_squared_error(y_val_actual, y_val_pred, squared=False)
    test_rmse = mean_squared_error(y_test_actual, y_test_pred, squared=False)

    # Print RMSE values
    print(f"Train RMSE: {train_rmse}")
    print(f"Validation RMSE: {val_rmse}")
    print(f"Test RMSE: {test_rmse}")

    # Save metrics to comparison.csv
    comparison_file_path = os.path.join(RESULT_FILES_DIR, "comparison.csv")
    df_metrics = pd.DataFrame({
        'Models': ["Model 7"],
        'Model Description': ["Gradient Boosting Regressor with Log-Target, Feature Engineering"],
        'Train error': [train_rmse],
        'Val error': [val_rmse],
        'Test error': [test_rmse]
    })
    if os.path.exists(comparison_file_path):
        df_existing_metrics = pd.read_csv(comparison_file_path)
        df_combined_metrics = pd.concat([df_existing_metrics, df_metrics], ignore_index=True)
    else:
        df_combined_metrics = df_metrics
    df_combined_metrics.to_csv(comparison_file_path, index=False)

    # Prepare results for output
    test_results = pd.DataFrame(X_test, columns=selected_features)
    test_results['Actual_Adjusted_Cost'] = y_test_actual.values
    test_results['Predicted_Adjusted_Cost'] = y_test_pred
    test_results['Error'] = test_results['Actual_Adjusted_Cost'] - test_results['Predicted_Adjusted_Cost']

    # Combine results with the original dataset's relevant columns
    test_indices = y_test.index
    test_results_full = pd.concat([df.loc[test_indices, :].reset_index(drop=True), test_results.reset_index(drop=True)], axis=1)

    # Save the detailed results to CSV
    test_results_full.to_csv(save_path, index=False)
    print(f"Detailed test results saved to: {save_path}")

# Main function to execute the process
def main():
    output_file_path = os.path.join(OUTPUT_FILES_DIR, 'model_7_result.csv')

    # Load and preprocess data
    X_train, X_val, X_test, y_train, y_val, y_test, selected_features, df = load_and_preprocess_data(RAW_FILE_PATH)

    # Train, validate, and test the model
    train_validate_test_model(X_train, X_val, X_test, y_train, y_val, y_test, selected_features, df, output_file_path)

# Execute the main function
if __name__ == "__main__":
    main()
