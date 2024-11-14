import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv('dynamic_pricing.csv')

# Calculate Revenue
df['Revenue'] = df['Number_of_Riders'] * df['Historical_Cost_of_Ride']

# Calculate Total Cost (assuming Historical Cost is the cost)
df['Total_Cost'] = df['Historical_Cost_of_Ride']  # or another cost calculation if applicable

# Calculate Profit
df['Total_Profit'] = df['Revenue'] - df['Total_Cost']

# Calculate Revenue_Per_Minute
df['Revenue_Per_Minute'] = df['Revenue'] / df['Expected_Ride_Duration']

# Calculate Profit_Per_Minute
df['Profit_Per_Minute'] = df['Total_Profit'] / df['Expected_Ride_Duration']

# Calculate Cost_Per_Transaction
df['Cost_Per_Transaction'] = df['Total_Cost'] / df['Number_of_Riders']

# Calculate Revenue_Per_Transaction
df['Revenue_Per_Transaction'] = df['Revenue'] / df['Number_of_Riders']

# Define the features (X) and target (y)
X = df[['Number_of_Riders', 'Number_of_Drivers', 'Location_Category', 'Customer_Loyalty_Status', 'Number_of_Past_Rides', 'Average_Ratings', 'Time_of_Booking', 'Vehicle_Type', 'Expected_Ride_Duration', 'Revenue_Per_Minute', 'Profit_Per_Minute', 'Cost_Per_Transaction', 'Revenue_Per_Transaction']]
y = df['Historical_Cost_of_Ride']

# Encoding categorical features using OneHotEncoder
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), ['Location_Category', 'Customer_Loyalty_Status', 'Time_of_Booking', 'Vehicle_Type']),  # Encode categorical columns
        ('num', 'passthrough', ['Number_of_Riders', 'Number_of_Drivers', 'Number_of_Past_Rides', 'Average_Ratings', 'Expected_Ride_Duration', 'Revenue_Per_Minute', 'Profit_Per_Minute', 'Cost_Per_Transaction', 'Revenue_Per_Transaction'])  # Keep numeric columns as is
    ])

# Create a pipeline for Linear Regression
linear_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Create a pipeline for Lasso Regression
lasso_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', Lasso(alpha=0.1))  # You can adjust alpha for regularization strength
])

# Train and predict using Linear Regression
linear_model.fit(X, y)
y_pred_linear = linear_model.predict(X)

# Train and predict using Lasso Regression
lasso_model.fit(X, y)
y_pred_lasso = lasso_model.predict(X)

# Calculate errors
error_lasso = y - y_pred_lasso
error_lr = y - y_pred_linear

# Create output DataFrame
output_df = pd.DataFrame({
    'Actual': y,
    'Predicted_Lasso': y_pred_lasso,
    'Predicted_LinearRegression': y_pred_linear,
    'Error_Lasso': error_lasso,
    'Error_LR': error_lr,
    'Revenue_Per_Minute': df['Revenue_Per_Minute'],
    'Profit_Per_Minute':df['Profit_Per_Minute'],
    'Cost_Per_Transaction':df['Cost_Per_Transaction'],
    'Cost_Per_Transaction':df['Cost_Per_Transaction']
})

# Save to CSV
output_df.to_csv('model_4_output.csv', index=False)
print("Output saved to 'model_4_output.csv'")
