import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error

# Load the data
data = pd.read_csv('dynamic_pricing.csv')

# Step 1: Feature Engineering
# Calculate 'Cost_Per_Minute'
data['Cost_Per_Minute'] = data['Historical_Cost_of_Ride'] / data['Expected_Ride_Duration']

# Add 'Location_Demand'
data['Location_Demand'] = data.groupby('Location_Category')['Number_of_Riders'].transform('count')
# Add 'Avg_Cost_Per_Vehicle_Type'
data['Avg_Cost_Per_Vehicle_Type'] = data.groupby('Vehicle_Type')['Historical_Cost_of_Ride'].transform('mean')

# Define features (X) and target (y)
X = data.drop(columns=['Historical_Cost_of_Ride'])
y = data['Historical_Cost_of_Ride']

# Separate numerical and categorical features
numerical_features = ['Number_of_Riders', 'Number_of_Drivers', 'Number_of_Past_Rides',
                      'Average_Ratings', 'Expected_Ride_Duration', 'Cost_Per_Minute',
                      'Location_Demand', 'Avg_Cost_Per_Vehicle_Type']
categorical_features = ['Location_Category', 'Customer_Loyalty_Status', 'Vehicle_Type']

# Train-test split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Step 2: Preprocessing with StandardScaler and OneHotEncoder
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# Step 3: Model Pipelines for Linear and Lasso Regression
# Linear Regression Pipeline
linear_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Lasso Regression Pipeline
lasso_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', Lasso(alpha=0.1))  # alpha is the regularization strength
])

# Step 4: Train models
linear_pipeline.fit(X_train, y_train)
lasso_pipeline.fit(X_train, y_train)

# Step 5: Predictions and Error Calculation
# Linear Regression
y_train_pred_linear = linear_pipeline.predict(X_train)
y_val_pred_linear = linear_pipeline.predict(X_val)
y_test_pred_linear = linear_pipeline.predict(X_test)

# Lasso Regression
y_train_pred_lasso = lasso_pipeline.predict(X_train)
y_val_pred_lasso = lasso_pipeline.predict(X_val)
y_test_pred_lasso = lasso_pipeline.predict(X_test)

# Calculate Mean Absolute Error for train, validation, and test sets
results = {
    'Model': ['Linear Regression', 'Lasso Regression'],
    'Train Error': [
        mean_absolute_error(y_train, y_train_pred_linear),
        mean_absolute_error(y_train, y_train_pred_lasso)
    ],
    'Validation Error': [
        mean_absolute_error(y_val, y_val_pred_linear),
        mean_absolute_error(y_val, y_val_pred_lasso)
    ],
    'Test Error': [
        mean_absolute_error(y_test, y_test_pred_linear),
        mean_absolute_error(y_test, y_test_pred_lasso)
    ]
}

# Step 6: Save results to CSV
results_data = pd.DataFrame(results)
results_data.to_csv('model_4_error.csv', index=False)

# Print results
print(results_data)
