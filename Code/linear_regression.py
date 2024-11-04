import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from constants import DATASET_PATH, TARGET_VARIABLE

# Load and preprocess dataset
data = pd.read_csv(DATASET_PATH)

# Convert `Average_Ratings` to categorical by binning
rating_bins = [0, 4.0, 4.5, 5.0]  # Define bin ranges
rating_labels = ['Low', 'Average', 'High']  # Define labels for each bin
data['Average_Ratings_Cat'] = pd.cut(data['Average_Ratings'], bins=rating_bins, labels=rating_labels)

# One-hot encode categorical features
data = pd.get_dummies(data, columns=['Location_Category', 'Customer_Loyalty_Status', 'Time_of_Booking', 'Vehicle_Type', 'Average_Ratings_Cat'], drop_first=True)

# Separate independent variables (X) and target variable (y)
X = data.drop(columns=[TARGET_VARIABLE])  # All columns except target
y = data[TARGET_VARIABLE]

# Split the data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Define a pipeline with scaling and linear regression
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Scaling features
    ('linear_regression', LinearRegression())  # Linear Regression model
])

# Train the model
pipeline.fit(X_train, y_train)

# Validate the model
y_val_pred = pipeline.predict(X_val)
val_rmse = mean_squared_error(y_val, y_val_pred, squared=False)
print(f'Validation RMSE: {val_rmse}')

# Test the model
y_test_pred = pipeline.predict(X_test)

# Calculate RMSE on test data
test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)
print(f'Test RMSE: {test_rmse}')

# Calculate error
error = y_test - y_test_pred

# Save the test dataset with predictions and error
results = X_test.copy()
results['Y_True'] = y_test
results['Y_Predicted'] = y_test_pred
results['Error'] = error

results.to_csv('C:/Users/viroc/Documents/Infosys Springboard Internship/Project/OptiPrice_Prognosticator_Infosys_Internship_Oct2024/Dataset/test_results_with_error.csv', index=False)
print("Test results with predictions and error saved to 'test_results_with_error.csv'")
