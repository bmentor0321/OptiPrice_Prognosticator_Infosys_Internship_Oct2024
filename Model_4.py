import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the dataset
df = pd.read_csv('dynamic_pricing.csv')

# Calculate Revenue
df['Revenue'] = df['Number_of_Riders'] * df['Historical_Cost_of_Ride']

# Calculate Total Cost (assuming Historical Cost is the cost)
df['Total_Cost'] = df['Historical_Cost_of_Ride']

# Calculate Profit
df['Total_Profit'] = df['Revenue'] - df['Total_Cost']

# Calculate Revenue Per Minute
df['Revenue_Per_Minute'] = df['Revenue'] / df['Expected_Ride_Duration']

# Calculate Profit Per Minute
df['Profit_Per_Minute'] = df['Total_Profit'] / df['Expected_Ride_Duration']

# Calculate Cost Per Transaction
df['Cost_Per_Transaction'] = df['Total_Cost'] / df['Number_of_Riders']

# Calculate Revenue Per Transaction
df['Revenue_Per_Transaction'] = df['Revenue'] / df['Number_of_Riders']

# Define the features (X) and target (y)
X = df[['Number_of_Riders', 'Number_of_Drivers', 'Location_Category', 'Customer_Loyalty_Status', 'Number_of_Past_Rides', 'Average_Ratings', 'Time_of_Booking', 'Vehicle_Type', 'Expected_Ride_Duration', 'Revenue_Per_Minute', 'Profit_Per_Minute', 'Cost_Per_Transaction', 'Revenue_Per_Transaction']]
y = df['Historical_Cost_of_Ride']

# Encoding categorical features using OneHotEncoder
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), ['Location_Category', 'Customer_Loyalty_Status', 'Time_of_Booking', 'Vehicle_Type']), 
        ('num', 'passthrough', ['Number_of_Riders', 'Number_of_Drivers', 'Number_of_Past_Rides', 'Average_Ratings', 'Expected_Ride_Duration', 'Revenue_Per_Minute', 'Profit_Per_Minute', 'Cost_Per_Transaction', 'Revenue_Per_Transaction'])
    ])

# Create a pipeline for Linear Regression
linear_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Create a pipeline for Lasso Regression
lasso_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', Lasso(alpha=0.1))
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
    'Profit_Per_Minute': df['Profit_Per_Minute'],
    'Cost_Per_Transaction': df['Cost_Per_Transaction'],
})

# Save to CSV
output_df.to_csv('Model_4_Errors.csv', index=False)
print("Output saved to 'Model_4_Errors.csv'")
