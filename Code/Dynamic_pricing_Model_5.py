# Importing Libraries
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder,StandardScaler,OrdinalEncoder
from sklearn.metrics import r2_score
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from Utils.constants import PROCESSED_DIR,RESULTS_DIR

# Load the dataset
input_file = os.path.join(PROCESSED_DIR, "Dynamic_pricing adjustment with new cost.csv")
df= pd.read_csv(input_file)

# Calculating total revenue through client current pricing methodology
total_Historical_cost=df['Historical_Cost_of_Ride'].sum()


                                            # Data Spliting and Model training


x=df[['Demand_Supply_Ratio','Location_Category','Average_Ratings','Customer_Loyalty_Status',
      'Number_of_Past_Rides','Time_of_Booking','Vehicle_Type','Expected_Ride_Duration']] #Feature column

y=df['New_cost'] #Target column

X_train_val, X_test, y_train_val, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

loyalty_order = ['Regular', 'Silver', 'Gold']
vehicle_type_order = ['Economy', 'Premium']

transformer= ColumnTransformer(transformers=[
    # Apply OneHotEncoder for Nominal columns
        ('location', OneHotEncoder(), ['Location_Category']),
        ('time', OneHotEncoder(), ['Time_of_Booking']),
        
        # Apply OrdinalEncoder for Ordinal columns with specific orders
        ('loyalty', OrdinalEncoder(categories=[loyalty_order]), ['Customer_Loyalty_Status']),
        ('vehicle', OrdinalEncoder(categories=[vehicle_type_order]), ['Vehicle_Type']),
        
        # Apply StandardScaler for Numerical columns
        ('scaler', StandardScaler(), ['Average_Ratings','Number_of_Past_Rides','Expected_Ride_Duration'])
    ],
    remainder='passthrough'
)

transformer.fit(X_train)

# Transform the training, validation, and test sets
X_train_trns = transformer.transform(X_train)
X_val_trns = transformer.transform(X_val)
X_test_trns = transformer.transform(X_test)

rf=RandomForestRegressor()

rf.fit(X_train_trns,y_train)



                                            # Model Evaluation & Check prediction

# Using RandomForest

# 1. Model evaluation on Train Data
y_train_pred_rf = rf.predict(X_train_trns)
train_rmse = np.sqrt(mean_squared_error(y_train,y_train_pred_rf))
train_r2 = r2_score(y_train,y_train_pred_rf)
print("Error of RandomForest Regression Model on train Data = ",train_rmse)
print("R2 score of RandomForest Regression on train Data = %.2f"%(train_r2))

# 2. Model evaluation on validation Data
y_val_pred_rf = rf.predict(X_val_trns)
val_rmse = np.sqrt(mean_squared_error(y_val,y_val_pred_rf))
val_r2 = r2_score(y_val,y_val_pred_rf)
print("Error of RandomForest Regression Model on validation Data = ", val_rmse)
print("R2 score of RandomForest Regression on validation Data = %.2f"%(val_r2))


# 3. Model evaluation on Test Data
y_test_pred_rf = rf.predict(X_test_trns)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred_rf))
test_r2 = r2_score(y_test, y_test_pred_rf)
print("Error of RandomForest Regression Model on test Data = ",test_rmse)
print("R2 score of RandomForest Regression on test data:", test_r2)


#using Linearregression
lr=LinearRegression()                                    
lr.fit(X_train_trns,y_train)



                                            # Model Evaluation & Check prediction

# 1. Model evaluation on Training Data
y_train_pred_lr = lr.predict(X_train_trns)
train_rmse_lr = np.sqrt(mean_squared_error(y_train,y_train_pred_lr))
train_r2_lr = r2_score(y_train,y_train_pred_lr)
print("Error of Linear Regression Model on Training Data  = ",train_rmse_lr)
print("R2 score of Linear Regression on Training Data = %.2f"%(train_r2_lr))

# 2. Model evaluation on validation Data
y_val_pred_lr = lr.predict(X_val_trns)
val_rmse_lr = np.sqrt(mean_squared_error(y_val,y_val_pred_lr))
val_r2_lr = r2_score(y_val,y_val_pred_lr)
print("Error of Linear Regression Model = ", val_rmse_lr)
print("R2 score of Linear Regression = %.2f"%(val_r2_lr))


# 3. Model evaluation on Test Data
y_test_pred_lr = lr.predict(X_test_trns)
test_rmse_lr = np.sqrt(mean_squared_error(y_test, y_test_pred_lr))
test_r2_lr = r2_score(y_test, y_test_pred_lr)
print("Error of Linear Regression Model on test Data = ", test_rmse_lr)
print("R2 score of Linear Regression Model on test data:", test_r2_lr)

# Calculating profitibily using dynamic approach
print("Revenue before dynamic approach ",total_Historical_cost)

revenue=sum(y_train_pred_rf) + sum(y_val_pred_rf) + sum(y_test_pred_rf)
print("Revenue after dynamic approach ",revenue)

profitibilty=revenue-total_Historical_cost
print("Increased Profitibilty after dynamic approach ",profitibilty/total_Historical_cost*100)
                                            
                                            # Save Test data, Prediction and Error
results_df = pd.DataFrame(X_test, columns=x.columns)
results_df['Actual'] = y_test
results_df['Predicted_RandomForest'] = y_test_pred_rf
results_df['Error_RF'] = y_test_pred_rf-y_test
results_df['Predicted_LinearRegression'] = y_test_pred_lr
results_df['Error_LR'] = y_test_pred_lr-y_test
output_file = os.path.join(RESULTS_DIR, "test_results by model_5.csv")
results_df.to_csv(output_file, index=False)

df_matrics =pd.DataFrame(columns=['Model','Model_Description','Train_Error','Val_Error','Test_Error'])
df_matrics.loc[len(df_matrics)]=['Model_5 (Random Forest)','Using numerics and categoricals columns + Feature Engineering ',train_rmse,val_rmse,test_rmse]
df_matrics.loc[len(df_matrics)]=['Model_5 (Linear Regression)','Using numerics and categoricals columns + Feature Engineering',train_rmse_lr, val_rmse_lr, test_rmse_lr]

output_file = os.path.join(RESULTS_DIR, "models_result_matrics.csv")
df_matrics.to_csv(output_file, mode='a', header=not os.path.exists(output_file), index=False)
