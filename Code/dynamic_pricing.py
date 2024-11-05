import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np
import constants as C


file_path = 'G:\\Spring board internship\\OptiPrice_Prognosticator_Infosys_Internship_Oct2024\\Data\\Raw\\dynamic_pricing.csv'
data = pd.read_csv(file_path)


X = data.drop(columns=['Historical_Cost_of_Ride'])
Y = data['Historical_Cost_of_Ride']


preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), C.NUMERIC_FEATURES),
        ('cat', OneHotEncoder(), C.CATEGORICAL_FEATURES)
    ]
)


pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])


X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.2, random_state=42)
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)


pipeline.fit(X_train, Y_train)


Y_train_pred = pipeline.predict(X_train)
train_mse = mean_squared_error(Y_train, Y_train_pred)
print(f'Training Mean Squared Error: {train_mse}')


Y_val_pred = pipeline.predict(X_val)
val_mse = mean_squared_error(Y_val, Y_val_pred)
print(f'Validation Mean Squared Error: {val_mse}')


Y_test_pred = pipeline.predict(X_test)
test_mse = mean_squared_error(Y_test, Y_test_pred)
print(f'Test Mean Squared Error: {test_mse}')


output_dir = 'G:\\Spring board internship\\OptiPrice_Prognosticator_Infosys_Internship_Oct2024\\Data\\Processed'
os.makedirs(output_dir, exist_ok=True)

results = X_test.copy()
results['Actual'] = Y_test
results['Predicted'] = Y_test_pred
results['Error'] = results['Actual'] - results['Predicted']

output_file_path = os.path.join(output_dir, 'dynamic_pricing_results_with_processing.csv')
try:
    results.to_csv(output_file_path, index=False)
    print("CSV file saved successfully at:", output_file_path)
except Exception as e:
    print("Error saving CSV:", e)
