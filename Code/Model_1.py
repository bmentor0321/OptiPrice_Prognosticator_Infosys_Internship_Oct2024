import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error, r2_score

# Load your data
data = pd.read_csv("dynamic_pricing.csv") 

# Define features (X) and target (y)
X = data[['Number_of_Riders', 'Number_of_Drivers', 'Number_of_Past_Rides', 'Average_Ratings', 'Expected_Ride_Duration']]
y = data['Historical_Cost_of_Ride']

# Split the data into 70% training, 15% validation, and 15% testing sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


# Train Linear Regression model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)

# Train Lasso model
lasso_model = Lasso(alpha=0.1)  # Adjust alpha as needed
lasso_model.fit(X_train, y_train)
y_pred_lasso = lasso_model.predict(X_test)

# Calculate errors
error_lasso = y_test - y_pred_lasso
error_lr = y_test - y_pred_linear

# Create DataFrame
output_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted_lasso': y_pred_lasso,
    'Predicted_LinearRegression': y_pred_linear,
    'Error_lasso': error_lasso,
    'Error_LR': error_lr
})

# Export to CSV
output_df.to_csv('Model_1_Results.csv', index=False)
