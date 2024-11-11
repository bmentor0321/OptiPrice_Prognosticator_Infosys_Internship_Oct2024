import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline

# Load the dataset
data = pd.read_csv('dynamic_pricing.csv')

# Select features and target variable
numerical_features = ['Number_of_Riders', 'Number_of_Drivers', 'Number_of_Past_Rides', 'Average_Ratings',
                      'Expected_Ride_Duration']
categorical_features = ['Location_Category', 'Customer_Loyalty_Status', 'Time_of_Booking',
                        'Vehicle_Type']
X = data[numerical_features + categorical_features]
y = data['Historical_Cost_of_Ride']

# Split the data into train, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Preprocessing: StandardScaler for numerical features, OneHotEncoder for categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Models to evaluate
models = {
    "Linear Regression": LinearRegression(),
    "Lasso": Lasso(),
    "Ridge": Ridge(),
    "SVR": SVR()
}

# Dictionary to store errors
results = {
    "Model": [],
    "Train Error": [],
    "Validation Error": [],
    "Test Error": []
}

# Train each model and calculate errors
for name, model in models.items():
    # Create a pipeline with preprocessing and model training
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

    # Fit the model on the training data
    pipeline.fit(X_train, y_train)

    # Predict on train, validation, and test sets
    y_train_pred = pipeline.predict(X_train)
    y_val_pred = pipeline.predict(X_val)
    y_test_pred = pipeline.predict(X_test)

    # Calculate RMSE for train, validation, and test sets
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    # Store the results
    results["Model"].append(name)
    results["Train Error"].append(train_rmse)
    results["Validation Error"].append(val_rmse)
    results["Test Error"].append(test_rmse)

# Convert results to DataFrame
results_data = pd.DataFrame(results)

# Save the results to a CSV file
results_data.to_csv('model3_errors.csv', index=False)

print("Model training complete and results saved to 'model3_errors.csv'")
