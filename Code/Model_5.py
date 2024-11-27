# Import necessary libraries
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from Utils.utils import get_mean
from Utils.constants import RAW_DIR ,RESULT_FILES_DIR, RAW_FILE_PATH, OUTPUT_FILES_DIR, LOYALTY_DISCOUNTS, BASE_FARE,PEAK_HOURS_MULTIPLIER, TRAFFIC_MULTIPLIER, PAST_RIDES_DISCOUNT, DURATION_MULTIPLIER, DURATION_THRESHOLD, HISTORICAL_COST_ADJUSTMENT_FACTOR, DISTANCE_THRESHOLD,output_file_name

os.makedirs(OUTPUT_FILES_DIR, exist_ok=True)

# Feature Engineering and Preprocessing
def load_and_preprocess_data(filepath):
    # Load the data
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()  # Clean column names

    # Feature Engineering: Create custom features
    df["Rider_Driver_Ratio"] = df["Number_of_Riders"] / (df["Number_of_Drivers"] + 1)
    df["Log_Ride_Duration"] = np.log(df["Expected_Ride_Duration"] + 1)

    # Define target variable (Adjusted_Historical_Cost_of_Ride)
    TARGET_VARIABLE = 'Historical_Cost_of_Ride'
    df['Adjusted_Historical_Cost_of_Ride'] = df[TARGET_VARIABLE]

    # Drop unnecessary columns
    df.drop(['Average_Ratings', 'Rating_Category', TARGET_VARIABLE], axis=1, inplace=True, errors='ignore')

    # One-hot encoding for categorical features
    combined_features = ['Customer_Loyalty_Status', 'Location_Category', 'Vehicle_Type', 'Time_of_Booking']
    df_encoded = pd.get_dummies(df, columns=combined_features)

    # Correlation Analysis
    correlation_matrix = df_encoded.corr()
    target_correlation = correlation_matrix['Adjusted_Historical_Cost_of_Ride'].sort_values(ascending=False)

    # Set a correlation threshold to filter features
    correlation_threshold = 0.1
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
    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Calculate RMSE for Train, Validation, and Test sets
    y_train_pred = model.predict(X_train)
    train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)
    
    y_val_pred = model.predict(X_val)
    val_rmse = mean_squared_error(y_val, y_val_pred, squared=False)

    y_test_pred = model.predict(X_test)
    test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)
    r2 = r2_score(y_test, y_test_pred)

    # Print RMSE values
    print(f"Train RMSE: {train_rmse}")
    print(f"Validation RMSE: {val_rmse}")
    print(f"Test RMSE: {test_rmse}")
    print(f"Test R2 Score: {r2}")

    # Save metrics to comparison.csv
    comparison_file_path = os.path.join(RESULT_FILES_DIR, "comparison.csv")
    df_metrics = pd.DataFrame({
        'Models': ["Model 5"],
        'Model Description': ["Numerical Features with Feature Engineering and Correlation-Based Selection + Standard Scaling + One-Hot Encoding for Categorical Features + Adjusted Historical Cost of Ride as Target"],
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
    test_results['Actual_Adjusted_Cost'] = y_test.values
    test_results['Predicted_Adjusted_Cost'] = y_test_pred
    test_results['Error'] = test_results['Actual_Adjusted_Cost'] - test_results['Predicted_Adjusted_Cost']

    # Combine results with the original dataset's relevant columns
    test_results_full = pd.concat([df.loc[X_test.index, :], test_results], axis=1)

    # Save the detailed results to CSV
    test_results_full.to_csv(save_path, index=False)
    print(f"Detailed test results saved to: {save_path}")

# Main function to execute the process
def main():
    output_file_path = os.path.join(OUTPUT_FILES_DIR, 'model_5_result.csv')

    # Load and preprocess data
    X_train, X_val, X_test, y_train, y_val, y_test, selected_features, df = load_and_preprocess_data(RAW_FILE_PATH)

    # Train, validate, and test the model
    train_validate_test_model(X_train, X_val, X_test, y_train, y_val, y_test, selected_features, df, output_file_path)

# Execute the main function
if __name__ == "__main__":
    main()