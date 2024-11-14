import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from constants import DATASET_PATH, TARGET_VARIABLE

# Load dataset
data = pd.read_csv(DATASET_PATH)

# Convert `Average_Ratings` to categorical by binning
rating_bins = [0, 4.0, 4.5, 5.0]  # Define bin ranges
rating_labels = ['Low', 'Average', 'High']  # Define labels for each bin
data['Average_Ratings_Cat'] = pd.cut(data['Average_Ratings'], bins=rating_bins, labels=rating_labels)

# Feature Engineering

#1-63.61
# Calculate Adjusted Cost
data['Adjusted_Ride_Cost'] = np.sqrt(data['Historical_Cost_of_Ride'] * data['Expected_Ride_Duration'])

#2-17.69
data['Cost_per_Minute'] = data['Historical_Cost_of_Ride'] / data['Expected_Ride_Duration']

#3-58.88
time_cost_map = {'Morning': 1.15, 'Afternoon': 1.15, 'Evening': 1.1, 'Night': 1.2}
data['Booking_Time_Cost_Impact'] = data['Time_of_Booking'].map(time_cost_map) * data['Historical_Cost_of_Ride']

#4-54.47
#loyalty_cost_map = {'Silver': 0.9, 'Regular': 1.0, 'Gold': 0.85}
#data['Loyalty_Cost_Impact'] = data['Customer_Loyalty_Status'].map(loyalty_cost_map) * data['Historical_Cost_of_Ride']

#5-46.18
#location_cost_map = {'Urban': 1.2, 'Suburban': 1.1, 'Rural': 0.9}
#data['Location_Cost_Multiplier'] = data['Location_Category'].map(location_cost_map)
#data['Location_Adjusted_Cost'] = data['Historical_Cost_of_Ride'] * data['Location_Cost_Multiplier']

#6-7.34
#data['Cost_per_Rider'] = data['Historical_Cost_of_Ride'] / data['Number_of_Riders']

#7-25.8
#data['Duration_Adjusted_Cost'] = data['Historical_Cost_of_Ride'] * (data['Expected_Ride_Duration'] / data['Expected_Ride_Duration'].mean())

#8-11.36
#data['Loyalty_Cost_Impact'] = data['Number_of_Past_Rides'] * data['Historical_Cost_of_Ride']

# One-hot encode categorical features
data = pd.get_dummies(data, columns=['Location_Category', 'Customer_Loyalty_Status', 'Time_of_Booking', 'Vehicle_Type', 'Average_Ratings_Cat'], drop_first=True)

# Separate independent variables (X) and target variable (y)
X = data.drop(columns=[TARGET_VARIABLE])
y = data[TARGET_VARIABLE]



# Split the data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Define a pipeline with scaling and linear regression
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('linear_regression', LinearRegression())
])

# Train the model
pipeline.fit(X_train, y_train)

# Calculate training error (RMSE)
y_train_pred = pipeline.predict(X_train)
train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)
print(f'Training RMSE: {train_rmse}')

# Validate the model
y_val_pred = pipeline.predict(X_val)
val_rmse = mean_squared_error(y_val, y_val_pred, squared=False)
print(f'Validation RMSE: {val_rmse}')

# Test the model
y_test_pred = pipeline.predict(X_test)
test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)
print(f'Test RMSE: {test_rmse}')

# Calculate error
error = y_test - y_test_pred

# Save the test dataset with predictions and error
results = X_test.copy()
results['Y_True'] = y_test
results['Y_Predicted'] = y_test_pred
results['Error'] = error

results.to_csv('C:/Users/viroc/Documents/Infosys Springboard Internship/Project/OptiPrice_Prognosticator_Infosys_Internship_Oct2024/Dataset/Model_4_test_results_with_error.csv', index=False)
print("Test results with predictions and error saved to 'Model_4_test_results_with_error.csv'")
