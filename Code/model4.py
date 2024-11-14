#!/usr/bin/env python
# coding: utf-8

# # Load the dataset
# 

# In[26]:
import numpy as np
import os
from Utils.constants import RAW_DIR,RESULTS_DIR,PROCESSED_DIR
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import joblib
file_path = os.path.join(RAW_DIR, "dynamic_pricing.csv")

# Load the dataset
df = pd.read_csv(file_path)
df


# In[ ]:





# In[27]:




data=df
location_factors = {'urban': 1.1, 'suburban': 1.0}
loyalty_discounts = {'Loyal': 0.95, 'Non-loyal': 1.0}
peak_hour_factors = {'peak': 1.1, 'off-peak': 1.0}

# Dynamic price adjustment function
def dynamic_price_adjustment(historical_cost, demand, supply, location, loyalty, peak_hour):
    # Get the respective factors for location, loyalty, and peak hour
    location_factor = location_factors.get(location, 1.0)
    loyalty_discount = loyalty_discounts.get(loyalty, 1.0)
    peak_hour_factor = peak_hour_factors.get(peak_hour, 1.0)
    
    # Calculate demand-supply adjustment
    demand_supply_adjustment = 1 + 0.2 * ((demand / supply > 1.2) - (demand / supply < 0.8))
    
    # Calculate adjusted price
    adjusted_price = historical_cost * demand_supply_adjustment * location_factor * loyalty_discount * peak_hour_factor
    return round(adjusted_price, 2)

# Apply the function row-wise to create a new column
data['Adjusted_Cost'] = data.apply(lambda row: dynamic_price_adjustment(
    row['Historical_Cost_of_Ride'],
    row['Number_of_Riders'],
    row['Number_of_Drivers'],
    row['Location_Category'],
    row['Customer_Loyalty_Status'],
    row['Time_of_Booking']
), axis=1)

# Display the updated DataFrame
data
scaler = MinMaxScaler()



# Applying feature engineering to add new features
data['demand_supply_ratio'] = data['Number_of_Riders'] / data['Number_of_Drivers']
data['Normalized_Historical_Cost'] = scaler.fit_transform(data[['Historical_Cost_of_Ride']])
data['cost_per_minute'] = data['Historical_Cost_of_Ride'] / data['Expected_Ride_Duration']

# Map peak/off-peak status
peak_hours = {'Night': 'peak', 'Evening': 'peak', 'Afternoon': 'off-peak'}
data['Booking_Peak_Status'] = data['Time_of_Booking'].map(peak_hours).fillna('off-peak')
data = pd.get_dummies(data, columns=['Booking_Peak_Status'], drop_first=True)

# Encode loyalty status and location-loyalty interaction
loyalty_mapping = {'Regular': 0, 'Silver': 1, 'Gold': 2}
data['Customer_Loyalty_Encoded'] = data['Customer_Loyalty_Status'].map(loyalty_mapping)
data['Location_Loyalty_Interaction'] = data['Location_Category'] + '_' + data['Customer_Loyalty_Status']
data = pd.get_dummies(data, columns=['Location_Loyalty_Interaction'], drop_first=True)

# Drop original columns not needed for model training
X = data.drop(columns=['Historical_Cost_of_Ride', 'Adjusted_Cost', 'Time_of_Booking', 'Customer_Loyalty_Status', 'Location_Category'])
y = data['Adjusted_Cost']
X = pd.get_dummies(X, drop_first=True)

# Normalize the independent variables

X_normalized = scaler.fit_transform(X)


df=data

# ## Step 2: Split the data into training, validation, and testing sets

# In[29]:


X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)  # 70% training
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)  # 15% validation, 15%


# 
# ## Step 3: Train the model

# In[30]:


model = LinearRegression()
model.fit(X_train, y_train)
y_train_pred = model.predict(X_train)
train_mae = mean_absolute_error(y_train, y_train_pred)
train_mse =mean_squared_error(y_train, y_train_pred)
print(f"Training MAE: {train_mae}, Training MSE: {train_mse}")


# ## Step 4: Validate the model

# In[31]:


y_val_pred = model.predict(X_val)
val_mae = mean_absolute_error(y_val, y_val_pred)
val_mse = mean_squared_error(y_val, y_val_pred)
print(f"Validation MAE: {val_mae}, Validation MSE: {val_mse}")



# ## Step 5: Test the model

# In[32]:


y_test_pred = model.predict(X_test)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_mse =mean_squared_error(y_test, y_test_pred)
print(f"Test MAE: {test_mae}, Test MSE: {test_mse}")


# ## Step 6: Calculate the error between Y and Y(Predicted) on the test set

# In[33]:


errors = y_test - y_test_pred
errors


# ## Step 7: Save Test dataset + Y variable + Y(predicted) + Error into a CSV file

# In[34]:


test_results = pd.DataFrame(X_test, columns=X.columns)
test_results['Y_True'] = y_test
test_results['Y_Predicted'] = y_test_pred
test_results['Error'] = errors

output_file = os.path.join(RESULTS_DIR, "Linear_Regression_result4.csv")
test_results.to_csv(output_file, index=False)

print(f"Output data saved to {output_file}")

df_metrics = pd.DataFrame(columns=['Model', "Model Description", "Train error", "Val error", "Test error"])

df_metrics.loc[len(df_metrics)] = ["Model4", "Feature engineering", train_mse, val_mse, test_mse]

if os.path.exists(os.path.join(RESULTS_DIR,"comparison.csv")):
    df_metrics_ = pd.read_csv(os.path.join(RESULTS_DIR,"comparison.csv"))

    df_metrics_ = pd.concat([df_metrics_, df_metrics])

df_metrics_.to_csv(os.path.join(RESULTS_DIR,"comparison.csv"), index=False)



