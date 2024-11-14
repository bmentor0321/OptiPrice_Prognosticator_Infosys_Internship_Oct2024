import os
from Utils.constants import RAW_DIR, PROCESSED_DIR
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.metrics import mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Load the dataset
input_file = os.path.join(RAW_DIR, "dynamic_pricing.csv")
data = pd.read_csv(input_file)

# Step 1: Enhanced Feature Engineering
data['Cost_Per_Minute'] = data['Historical_Cost_of_Ride'] / data['Expected_Ride_Duration']
data['Location_Demand'] = data.groupby('Location_Category')['Number_of_Riders'].transform('count')
data['Avg_Cost_Per_Vehicle_Type'] = data.groupby('Vehicle_Type')['Historical_Cost_of_Ride'].transform('mean')

# Adding interaction and polynomial features
data['Demand_Cost_Interaction'] = data['Location_Demand'] * data['Cost_Per_Minute']

# Define features (X) and target (y)
X = data.drop(columns=['Historical_Cost_of_Ride'])
y = data['Historical_Cost_of_Ride']

# Separate numerical and categorical features
numerical_features = ['Number_of_Riders', 'Number_of_Drivers', 'Number_of_Past_Rides',
                      'Average_Ratings', 'Expected_Ride_Duration', 'Cost_Per_Minute',
                      'Location_Demand', 'Avg_Cost_Per_Vehicle_Type', 'Demand_Cost_Interaction']
categorical_features = ['Location_Category', 'Customer_Loyalty_Status', 'Vehicle_Type']

# Train-test split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Step 2: Preprocessing with StandardScaler, OneHotEncoder, and PolynomialFeatures
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('scaler', StandardScaler()),
            ('poly', PolynomialFeatures(degree=2, include_bias=False))
        ]), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# Step 3: Model Pipelines with Cross-Validation and Hyperparameter Tuning for Lasso
# Define model pipelines with GridSearchCV for Lasso and Ridge
models = {
    'Linear Regression': Pipeline(steps=[('preprocessor', preprocessor), ('regressor', LinearRegression())]),
    'Lasso Regression': Pipeline(steps=[('preprocessor', preprocessor), ('regressor', Lasso())]),
    'Ridge Regression': Pipeline(steps=[('preprocessor', preprocessor), ('regressor', Ridge())])
}

# Define hyperparameters to tune
param_grid = {
    'Lasso Regression': {'regressor__alpha': [0.01, 0.1, 1, 10]},
    'Ridge Regression': {'regressor__alpha': [0.01, 0.1, 1, 10]}
}

# Initialize results dictionary
results = {'Model': [], 'Train Error': [], 'Validation Error': [], 'Test Error': []}

# Step 4: Train and Evaluate Models with Cross-Validation
for model_name, pipeline in models.items():
    if model_name in param_grid:
        # Perform GridSearchCV for models with hyperparameters
        grid_search = GridSearchCV(pipeline, param_grid[model_name], cv=5, scoring='neg_mean_absolute_error')
        grid_search.fit(X_train, y_train)
        best_pipeline = grid_search.best_estimator_
    else:
        # Directly fit models without hyperparameters
        best_pipeline = pipeline.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = best_pipeline.predict(X_train)
    y_val_pred = best_pipeline.predict(X_val)
    y_test_pred = best_pipeline.predict(X_test)

    # Calculate Mean Absolute Error for train, validation, and test sets
    results['Model'].append(model_name)
    results['Train Error'].append(mean_absolute_error(y_train, y_train_pred))
    results['Validation Error'].append(mean_absolute_error(y_val, y_val_pred))
    results['Test Error'].append(mean_absolute_error(y_test, y_test_pred))

# Convert results to DataFrame
results_data = pd.DataFrame(results)

# Save the results to a CSV file
output_file = os.path.join(PROCESSED_DIR, "model4_errors.csv")
results_data.to_csv(output_file, index=False)

# Print results
print(results_data)
