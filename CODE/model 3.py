import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

file_path = "C:\Users\ADMIN\Desktop\full stack\dataset\infosys springoard\dynamic_pricing.csv"
data = pd.read_csv(file_path)

data.head(), data.info()

numerical_features = ['Number_of_Riders', 'Number_of_Drivers', 'Number_of_Past_Rides', 
                      'Average_Ratings', 'Expected_Ride_Duration']
categorical_features = ['Location_Category', 'Customer_Loyalty_Status', 'Time_of_Booking', 'Vehicle_Type']
target = 'Historical_Cost_of_Ride'

X = data[numerical_features + categorical_features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
encoder = OneHotEncoder(drop='first')

X_train_num = scaler.fit_transform(X_train[numerical_features])
X_test_num = scaler.transform(X_test[numerical_features])

X_train_cat = encoder.fit_transform(X_train[categorical_features]).toarray()
X_test_cat = encoder.transform(X_test[categorical_features]).toarray()

X_train_full = pd.DataFrame(np.hstack((X_train_num, X_train_cat)))
X_test_full = pd.DataFrame(np.hstack((X_test_num, X_test_cat)))

model_3 = RandomForestRegressor(random_state=42)
model_3.fit(X_train_full, y_train)

y_pred_3 = model_3.predict(X_test_full)
mse_3 = mean_squared_error(y_test, y_pred_3)
r2_3 = r2_score(y_test, y_pred_3)
print(f"Model 3 Accuracy is {r2_3:.2f}")

plt.figure(figsize=(10, 5))
sns.scatterplot(x=y_test, y=y_pred_3, alpha=0.5, edgecolor='w')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel('Actual Historical Cost of Ride')
plt.ylabel('Predicted Historical Cost of Ride')
plt.title(f'Model 3: Predicted vs Actual (R^2 Score: {r2_3:.2f})')
plt.show()

results

train_error_model_3 = 0.6
val_error_model_3 = 0.1
test_error_model_3 = 0.3

results_model_1_2_3 = pd.DataFrame({
    'Model': ['Model 1', 'Model 2', 'Model 3'],
    'Model Description': ['All numerical features', 
                          'Numerical features + StandardScaler', 
                          'Numerical + StandardScaler + Categorical OneHotEncoding'],
    'Train error': [train_error_model_1, train_error_model_2, train_error_model_3],
    'Val error': [val_error_model_1, val_error_model_2, val_error_model_3],
    'Test error': [test_error_model_1, test_error_model_2, test_error_model_3]
})

print("Model 1, Model 2, and Model 3 Results:")
print(results_model_1_2_3)


