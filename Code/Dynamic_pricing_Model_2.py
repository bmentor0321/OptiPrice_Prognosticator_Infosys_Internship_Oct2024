# Importing Libraries
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from Utils.constants import RESULTS_DIR,RAW_DIR

# Load the dataset
input_file = os.path.join(RAW_DIR, "dynamic_pricing.csv")
df= pd.read_csv(input_file)



                                            # Data Spliting and Model training



x=df[['Number_of_Riders','Number_of_Drivers','Number_of_Past_Rides','Average_Ratings','Expected_Ride_Duration']] #Feature column numericals only
y=df['Historical_Cost_of_Ride'] #Target column

X_train_val, X_test, y_train_val, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

rf=RandomForestRegressor()                                    
rf.fit(X_train_scaled,y_train)

y_pred_rf=rf.predict(X_test_scaled)



                                            # Model Evaluation & Check prediction

# Using RandomForest

# 1. Model evaluation on Train Data
y_train_pred_rf = rf.predict(X_train_scaled)
train_rmse = np.sqrt(mean_squared_error(y_train,y_train_pred_rf))
train_r2 = r2_score(y_train,y_train_pred_rf)
print("Error of RandomForest Regression Model on train Data = ",train_rmse)
print("R2 score of RandomForest Regression on train Data = %.2f"%(train_r2))

# 2. Model evaluation on validation Data
y_val_pred_rf = rf.predict(X_val_scaled)
val_rmse = np.sqrt(mean_squared_error(y_val,y_val_pred_rf))
val_r2 = r2_score(y_val,y_val_pred_rf)
print("Error of RandomForest Regression Model on validation Data = ", val_rmse)
print("R2 score of RandomForest Regression on validation Data = %.2f"%(val_r2))


# 3. Model evaluation on Test Data
y_test_pred_rf = rf.predict(X_test_scaled)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred_rf))
test_r2 = r2_score(y_test, y_test_pred_rf)
print("Error of RandomForest Regression Model on test Data = ",test_rmse)
print("R2 score of RandomForest Regression on test data:", test_r2)


#using Linearregression
lr=LinearRegression()                                    
lr.fit(X_train_scaled,y_train)

y_pred_lr=lr.predict(X_test_scaled)



                                            # Model Evaluation & Check prediction

# 1. Model evaluation on Training Data
y_train_pred_lr = lr.predict(X_train_scaled)
train_rmse_lr = np.sqrt(mean_squared_error(y_train,y_train_pred_lr))
train_r2_lr = r2_score(y_train,y_train_pred_lr)
print("Error of Linear Regression Model on Training Data  = ",train_rmse_lr)
print("R2 score of Linear Regression on Training Data = %.2f"%(train_r2_lr))

# 2. Model evaluation on validation Data
y_val_pred_lr = lr.predict(X_val_scaled)
val_rmse_lr = np.sqrt(mean_squared_error(y_val,y_val_pred_lr))
val_r2_lr = r2_score(y_val,y_val_pred_lr)
print("Error of Linear Regression Model = ", val_rmse_lr)
print("R2 score of Linear Regression = %.2f"%(val_r2_lr))


# 3. Model evaluation on Test Data
y_test_pred_lr = lr.predict(X_test_scaled)
test_rmse_lr = np.sqrt(mean_squared_error(y_test, y_test_pred_lr))
test_r2_lr = r2_score(y_test, y_test_pred_lr)
print("Error of Linear Regression Model on test Data = ", test_rmse_lr)
print("R2 score of Linear Regression Model on test data:", test_r2_lr)


                                            
                                            # Save Test data, Prediction and Error
results_df = pd.DataFrame(X_test, columns=x.columns)
results_df['Actual'] = y_test
results_df['Predicted_RandomForest'] = y_pred_rf
results_df['Predicted_LinearRegression'] = y_pred_lr
results_df['Error_RF'] = y_pred_rf-y_test
results_df['Error_LR'] = y_pred_lr-y_test
output_file = os.path.join(RESULTS_DIR, "test_results by model_2.csv")
results_df.to_csv(output_file, index=False)

df_matrics =pd.DataFrame(columns=['Model','Model_Description','Train_Error','Val_Error','Test_Error'])
df_matrics.loc[len(df_matrics)]=['Model_2 (Random Forest)','Using only numerics columns and applying Standard Scaling',train_rmse,val_rmse,test_rmse]
df_matrics.loc[len(df_matrics)]=['Model_2 (Linear Regression)','Using only numerics columns and applying Standard Scaling',train_rmse_lr, val_rmse_lr, test_rmse_lr]

output_file = os.path.join(RESULTS_DIR, "models_result_matrics.csv")
df_matrics.to_csv(output_file, mode='a', header=not os.path.exists(output_file), index=False)