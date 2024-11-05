# Importing Libraries
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_percentage_error,mean_squared_error,root_mean_squared_error
from Utils.constants import PROCESSED_DIR,RESULTS_DIR

# Load the dataset
input_file = os.path.join(PROCESSED_DIR, "Dynamic_pricing adjustment with new cost.csv")
df= pd.read_csv(input_file)



                                            # Data Spliting and Model training



x=df[['Demand_Supply_Ratio','Location_Category','Time_of_Booking','Vehicle_Type','Expected_Ride_Duration']] #Train column
y=df['New_cost'] #Target column

X_train_val, X_test, y_train_val, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

ohe = OneHotEncoder()
ohe.fit(x[['Location_Category','Time_of_Booking','Vehicle_Type']])
cat=ohe.categories_

column_trans = make_column_transformer((OneHotEncoder(categories=cat),
                                        ['Location_Category','Time_of_Booking','Vehicle_Type']),
                                        remainder='passthrough')    

rf=RandomForestRegressor()                                    
pipe=make_pipeline(column_trans,rf)
pipe.fit(X_train,y_train)

y_pred_rf=pipe.predict(X_test)



                                            # Model Evaluation & Check prediction

# Using RandomForest

# 1. Model evaluation on Train Data
y_train_pred_rf = pipe.predict(X_train)
train_mape = mean_absolute_percentage_error(y_train,y_train_pred_rf)
train_r2 = r2_score(y_train,y_train_pred_rf)
print("Error of RandomForest Regression Model on train Data = %.2f"%(train_mape*100),'%')
print("Accuracy of RandomForest Regression Model on train Data = %.2f"%((1 - train_mape)*100),'%')
print("R2 score of RandomForest Regression on train Data = %.2f"%(train_r2))

# 2. Model evaluation on validation Data
y_val_pred_rf = pipe.predict(X_val)
val_mape = mean_absolute_percentage_error(y_val,y_val_pred_rf)
val_r2 = r2_score(y_val,y_val_pred_rf)
print("Error of RandomForest Regression Model on validation Data = %.2f"%(val_mape*100),'%')
print("Accuracy of RandomForest Regression Model on validation Data = %.2f"%((1 - val_mape)*100),'%')
print("R2 score of RandomForest Regression on validation Data = %.2f"%(val_r2))


# 3. Model evaluation on Test Data
y_test_pred_rf = pipe.predict(X_test)
test_mape = mean_absolute_percentage_error(y_test, y_test_pred_rf)
test_r2 = r2_score(y_test, y_test_pred_rf)
print("Error of RandomForest Regression Model on test Data = %.2f"%(test_mape*100),'%')
print("R2 score of RandomForest Regression on test data:", test_r2)


#using Linearregression
lr=LinearRegression()                                    
pipe=make_pipeline(column_trans,lr)
pipe.fit(X_train,y_train)

y_pred_lr=pipe.predict(X_test)



                                            # Model Evaluation & Check prediction

# 1. Model evaluation on Training Data
y_train_pred_lr = pipe.predict(X_train)
train_mape_lr = mean_absolute_percentage_error(y_train,y_train_pred_lr)
train_r2_lr = r2_score(y_train,y_train_pred_lr)
print("Error of Linear Regression Model on validation Data  = %.2f"%(train_mape_lr*100),'%')
print("Accuracy of Linear Regression Model on validation Data = %.2f"%((1 - train_mape_lr)*100),'%')
print("R2 score of Linear Regression on validation Data = %.2f"%(train_r2_lr))

# 2. Model evaluation on validation Data
y_val_pred_lr = pipe.predict(X_val)
val_mape_lr = mean_absolute_percentage_error(y_val,y_val_pred_lr)
val_r2_lr = r2_score(y_val,y_val_pred_lr)
print("Error of Linear Regression Model = %.2f"%(val_mape_lr*100),'%')
print("Accuracy of Linear Regression Model = %.2f"%((1 - val_mape_lr)*100),'%')
print("R2 score of Linear Regression = %.2f"%(val_r2_lr))


# 3. Model evaluation on Test Data
y_test_pred_lr = pipe.predict(X_test)
test_mape_lr = mean_absolute_percentage_error(y_test, y_test_pred_lr)
test_r2_lr = r2_score(y_test, y_test_pred_lr)
print("Error of Linear Regression Model on test Data = %.2f"%(test_mape*100),'%')
print("R2 score of Linear Regression Model on test data:", test_r2_lr)


                                            
                                            # Save Test data, Prediction and Error
results_df = pd.DataFrame(X_test, columns=x.columns)
results_df['Actual'] = y_test
results_df['Predicted_RandomForest'] = y_pred_rf
results_df['Predicted_LinearRegression'] = y_pred_lr
results_df['Error_RF'] = y_pred_rf-y_test
results_df['Error_LR'] = y_pred_lr-y_test
output_file = os.path.join(RESULTS_DIR, "test_results.csv")
results_df.to_csv(output_file, index=False)

df_matrics =pd.DataFrame(columns=['Model','Model_Description','Train_Error','Val_Error','Test_Error'])
df_matrics.loc[len(df_matrics)]=['Random Forest','Using numerics and categoricals columns',train_mape,val_mape,test_mape]
df_matrics.loc[len(df_matrics)]=['Linear Regression','Using numerics and categoricals columns',train_mape_lr,val_mape_lr,test_mape_lr]

output_file = os.path.join(RESULTS_DIR, "model_result_matrics.csv")
df_matrics.to_csv(output_file, index=False)