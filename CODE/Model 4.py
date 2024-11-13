import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer

# Step 1: Load the dataset
data = pd.read_csv('dynamic_pricing.csv')

# Step 2: Feature Engineering
# Calculating 'Cost_Per_Minute'
data['Cost_Per_Minute'] = data['Historical_Cost_of_Ride'] / data['Expected_Ride_Duration']

# Step 3: Define features and target variable
X = data.drop(columns=['Historical_Cost_of_Ride'])
y = data['Historical_Cost_of_Ride']

# Separate numerical and categorical features
numerical_features = ['Number_of_Riders', 'Number_of_Drivers', 'Number_of_Past_Rides',
                      'Average_Ratings', 'Expected_Ride_Duration', 'Cost_Per_Minute']
if 'Cost_Per_Mile' in data.columns:
    numerical_features.append('Cost_Per_Mile')

categorical_features = ['Location_Category', 'Customer_Loyalty_Status', 'Time_of_Booking', 'Vehicle_Type']

# Step 4: Preprocessing Pipeline
# Numerical features: impute missing values and scale
# Categorical features: impute missing values and one-hot encode
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]), numerical_features),

        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(drop='first'))
        ]), categorical_features)
    ])

# Step 5: Train-Test Split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Step 6: Modeling and Training
# Linear Regression model pipeline
lin_reg_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])

# Train the Linear Regression model
lin_reg_pipeline.fit(X_train, y_train)

# Lasso Regression model pipeline
lasso_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', Lasso(alpha=0.1))
])

# Train the Lasso Regression model
lasso_pipeline.fit(X_train, y_train)


# Step 7: Evaluate Models and Save Results
# Function to calculate errors
def get_errors(model_pipeline, X_train, X_val, X_test, y_train, y_val, y_test):
    errors = {}
    errors['Train Error'] = mean_squared_error(y_train, model_pipeline.predict(X_train), squared=False)
    errors['Validation Error'] = mean_squared_error(y_val, model_pipeline.predict(X_val), squared=False)
    errors['Test Error'] = mean_squared_error(y_test, model_pipeline.predict(X_test), squared=False)
    return errors


# Get errors for both models
lin_reg_errors = get_errors(lin_reg_pipeline, X_train, X_val, X_test, y_train, y_val, y_test)
lasso_errors = get_errors(lasso_pipeline, X_train, X_val, X_test, y_train, y_val, y_test)

# Combine errors into a DataFrame
errors_data = pd.DataFrame({
    'Linear Regression': lin_reg_errors,
    'Lasso Regression': lasso_errors
}).T

# Step 8: Save errors to CSV
errors_data.to_csv('model4_errors.csv')

print("Training, validation, and test errors saved in 'model4_errors.csv'")
