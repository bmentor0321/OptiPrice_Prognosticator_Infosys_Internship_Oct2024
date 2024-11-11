import pandas as pd
import os
from sklearn.model_selection import train_test_split  # For splitting data into training, validation, and test sets
from sklearn.linear_model import LinearRegression  # For performing linear regression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error  # To evaluate the model
from sklearn.pipeline import Pipeline  # For creating a pipeline of transformations and model
from sklearn.compose import ColumnTransformer  # To apply different transformations to different feature types
from sklearn.preprocessing import StandardScaler, OneHotEncoder  # For scaling numeric features and encoding categorical ones
from sklearn.decomposition import PCA  # For dimensionality reduction
import numpy as np
import constants as C  # To access feature names and target column from constants.py

# Loading the dataset
file_path = 'G:\\Spring board internship\\OptiPrice_Prognosticator_Infosys_Internship_Oct2024\\Data\\Processed\\dynamic_pricing_results_with_pca.csv'
data = pd.read_csv(file_path)  # Reading the dataset into a DataFrame

# Selecting features and target
X = data[C.NUMERIC_FEATURES + C.CATEGORICAL_FEATURES]  # Combining numeric and categorical features
Y = data[C.TARGET_COLUMN]  # The target column for prediction

# Defining preprocessing steps
# I'm using `ColumnTransformer` to apply different preprocessing techniques to numeric and categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), C.NUMERIC_FEATURES),  # Scaling numeric features
        ('cat', OneHotEncoder(), C.CATEGORICAL_FEATURES)  # One-hot encoding categorical features
    ]
)

# Creating a pipeline for the whole workflow
# First, we preprocess the data, then apply PCA for dimensionality reduction, and finally fit a Linear Regression model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),  # Apply preprocessing to features
    ('pca', PCA(n_components=0.95)),  # Reduce dimensionality while retaining 95% of the variance
    ('model', LinearRegression())  # Using linear regression to predict
])

# Splitting the dataset into training, validation, and test sets
X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.2, random_state=42)  # Splitting into 80% training and 20% temporary
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)  # Splitting the temporary 20% into equal validation and test sets

# Training the model
# Now, I’m fitting the pipeline with the training data
pipeline.fit(X_train, Y_train)

# Evaluating model performance on training data
Y_train_pred = pipeline.predict(X_train)  # Making predictions on the training set
train_rmse = np.sqrt(mean_squared_error(Y_train, Y_train_pred))  # Calculating RMSE (Root Mean Squared Error)
train_r2 = r2_score(Y_train, Y_train_pred)  # Calculating R² (coefficient of determination)
train_mape = mean_absolute_percentage_error(Y_train, Y_train_pred)  # Calculating MAPE (Mean Absolute Percentage Error)

# Printing out the performance metrics for training data
print(f'Training RMSE: {train_rmse}')  # RMSE value for training
print(f'Training R²: {train_r2}')  # R² value for training
print(f'Training MAPE: {train_mape * 100:.2f}%')  # MAPE for training data

# Evaluating model performance on validation data
Y_val_pred = pipeline.predict(X_val)  # Making predictions on the validation set
val_rmse = np.sqrt(mean_squared_error(Y_val, Y_val_pred))  # RMSE for validation set
val_r2 = r2_score(Y_val, Y_val_pred)  # R² for validation set
val_mape = mean_absolute_percentage_error(Y_val, Y_val_pred)  # MAPE for validation set

# Printing out the performance metrics for validation data
print(f'Validation RMSE: {val_rmse}')  # RMSE value for validation
print(f'Validation R²: {val_r2}')  # R² value for validation
print(f'Validation MAPE: {val_mape * 100:.2f}%')  # MAPE for validation

# Evaluating model performance on test data
Y_test_pred = pipeline.predict(X_test)  # Making predictions on the test set
test_rmse = np.sqrt(mean_squared_error(Y_test, Y_test_pred))  # RMSE for test set
test_r2 = r2_score(Y_test, Y_test_pred)  # R² for test set
test_mape = mean_absolute_percentage_error(Y_test, Y_test_pred)  # MAPE for test set

# Printing out the performance metrics for test data
print(f'Test RMSE: {test_rmse}')  # RMSE value for test
print(f'Test R²: {test_r2}')  # R² value for test
print(f'Test MAPE: {test_mape * 100:.2f}%')  # MAPE for test data

# Saving the results
# Saving predictions and actual values for the test set along with the error
output_dir = 'G:\\Spring board internship\\OptiPrice_Prognosticator_Infosys_Internship_Oct2024\\Data\\Processed'
os.makedirs(output_dir, exist_ok=True)  # Creating the directory if it doesn't exist

# Creating a DataFrame with the results: actual values, predicted values, and the error
results = X_test.copy()  # Copying the test data
results['Actual'] = Y_test  # Adding actual target values to the results
results['Predicted'] = Y_test_pred  # Adding predicted values to the results
results['Error'] = results['Actual'] - results['Predicted']  # Calculating the error (difference between actual and predicted)

# Saving the results to a CSV file
output_file_path = os.path.join(output_dir, 'dynamic_pricing_change_of_y_vairable_results.csv')
try:
    results.to_csv(output_file_path, index=False)  # Writing results to CSV
    print("CSV saved successfully at:", output_file_path)  # Success message
except Exception as e:
    print("Error saving CSV:", e)  # Error message in case of failure
