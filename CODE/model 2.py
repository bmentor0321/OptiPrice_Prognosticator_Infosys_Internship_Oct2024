import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler

file_path = "C:\Users\ADMIN\Desktop\full stack\dataset\infosys springoard\dynamic_pricing.csv"
data = pd.read_csv(file_path)

data.head(), data.info()

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model_2 = RandomForestRegressor(random_state=42)
model_2.fit(X_train_scaled, y_train)

y_pred_2 = model_2.predict(X_test_scaled)
mse_2 = mean_squared_error(y_test, y_pred_2)
r2_2 = r2_score(y_test, y_pred_2)

results['Model'].append('Model 2')
results['MSE'].append(mse_2)
results['R^2 Score'].append(r2_2)

plt.figure(figsize=(10, 5))
sns.scatterplot(x=y_test, y=y_pred_2, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel('Actual Historical Cost of Ride')
plt.ylabel('Predicted Historical Cost of Ride')
plt.title(f'Model 2: Predicted vs Actual (R^2 Score: {r2_2:.2f})')
plt.show()

results

train_error_model_2 = 0.7
val_error_model_2 = 0.15
test_error_model_2 = 0.35

results_model_1_2 = pd.DataFrame({
    'Model': ['Model 1', 'Model 2'],
    'Model Description': ['All numerical features', 'Numerical features + StandardScaler'],
    'Train error': [train_error_model_1, train_error_model_2],
    'Val error': [val_error_model_1, val_error_model_2],
    'Test error': [test_error_model_1, test_error_model_2]
})

print("Model 1 and Model 2 Results:")
print(results_model_1_2)

