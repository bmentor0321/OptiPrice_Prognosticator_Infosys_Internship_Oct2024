import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from Utils.constants import RAW_DIR, file_name, RESULTS_DIR, output_file_name, processed_data_dir
from Utils.utils import get_mean

# File paths
NEW_DATAPATH = r'D:\Rishi\Data\Raw\dynamic_pricing.csv'  
df = data = RAW_DIR = pd.read_csv(os.path.join(RAW_DIR, file_name))
TARGET_VARIABLE = 'Historical_Cost_of_Ride'

# Feature Engineering and Preprocessing
def load_and_preprocess_data(filepath):
    # Load the data
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()  # Clean column names

    # Feature Engineering: Create custom features
    df["Rider_Driver_Ratio"] = df["Number_of_Riders"] / (df["Number_of_Drivers"] + 1)
    df["Log_Ride_Duration"] = np.log(df["Expected_Ride_Duration"] + 1)

    # Define target variable (Adjusted_Historical_Cost_of_Ride)
    df['Adjusted_Historical_Cost_of_Ride'] = df[TARGET_VARIABLE]

    # Drop unnecessary columns
    df.drop(['Average_Ratings', 'Rating_Category', TARGET_VARIABLE], axis=1, inplace=True, errors='ignore')

    # One-hot encoding for categorical features
    combined_features = ['Customer_Loyalty_Status', 'Location_Category', 'Vehicle_Type', 'Time_of_Booking']
    df_encoded = pd.get_dummies(df, columns=combined_features)

    # ---------- CORRELATION ANALYSIS ---------- #
    # Calculate correlation with the target variable
    correlation_matrix = df_encoded.corr()
    target_correlation = correlation_matrix['Adjusted_Historical_Cost_of_Ride'].sort_values(ascending=False)

    # Set a correlation threshold (e.g., 0.1) to filter features
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

    # Validate the model
    y_pred_val = model.predict(X_val)
    val_rmse = mean_squared_error(y_val, y_pred_val, squared=False)
    print(f"Validation RMSE: {val_rmse}")

    # Test the model
    y_pred_test = model.predict(X_test)
    test_rmse = mean_squared_error(y_test, y_pred_test, squared=False)
    r2 = r2_score(y_test, y_pred_test)
    print(f"Test RMSE: {test_rmse}")
    print(f"Test R2 Score: {r2}")

    # Prepare results for output (using the original data)
    test_results = pd.DataFrame(X_test, columns=selected_features)

    # Include all original columns from the dataset (e.g., the ones you dropped)
    test_results['Actual_Adjusted_Cost'] = y_test.values
    test_results['Predicted_Adjusted_Cost'] = y_pred_test
    test_results['Error'] = test_results['Actual_Adjusted_Cost'] - test_results['Predicted_Adjusted_Cost']

    # Combine the results with the original dataframe's relevant columns
    test_results_full = pd.concat([df.loc[X_test.index, :], test_results], axis=1)

    # Save the results to CSV
    test_results_full.to_csv(save_path, index=False)
    print(f"Results saved to: {save_path}")

# Main function to execute the process
def main():
    # File paths
    RESULTS_DIR = r'D:\Rishi\Data\Results'
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_file_path = os.path.join(RESULTS_DIR, 'model_5_results.csv')

    # Load and preprocess data
    X_train, X_val, X_test, y_train, y_val, y_test, selected_features, df = load_and_preprocess_data(NEW_DATAPATH)

    # Train, validate, and test the model
    train_validate_test_model(X_train, X_val, X_test, y_train, y_val, y_test, selected_features, df, output_file_path)

# Execute the main function
if __name__ == "__main__":
    main()
