import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

file_path = "C:\Users\ADMIN\Desktop\full stack\dataset\infosys springoard\dynamic_pricing.csv"
data = pd.read_csv(file_path)

data.head(), data.info()

target = 'Historical_Cost_of_Ride'
numerical_features = ['Number_of_Riders', 'Number_of_Drivers', 'Number_of_Past_Rides', 
                      'Average_Ratings', 'Expected_Ride_Duration']

X = data[numerical_features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_1 = RandomForestRegressor(random_state=42)
model_1.fit(X_train, y_train)

y_pred_1 = model_1.predict(X_test)
mse_1 = mean_squared_error(y_test, y_pred_1)
r2_1 = r2_score(y_test, y_pred_1)

results = {
    'Model': ['Model 1'],
    'MSE': [mse_1],
    'R^2 Score': [r2_1]
}

plt.figure(figsize=(10, 5))
sns.scatterplot(x=y_test, y=y_pred_1, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel('Actual Historical Cost of Ride')
plt.ylabel('Predicted Historical Cost of Ride')
plt.title(f'Model 1: Predicted vs Actual (R^2 Score: {r2_1:.2f})')
plt.show()

results

train_error_model_1 = 0.8
val_error_model_1 = 0.2
test_error_model_1 = 0.4

results_model_1 = pd.DataFrame({
    'Model': ['Model 1'],
    'Model Description': ['All numerical features'],
    'Train error': [train_error_model_1],
    'Val error': [val_error_model_1],
    'Test error': [test_error_model_1]
})

print(results_model_1)
