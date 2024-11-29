# Import necessary libraries
#importing required models
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score

# Define file paths
RAW_DIR = 'Data/Raw'
PROCESSED_DIR = 'Data/Processed'
input_file = os.path.join(RAW_DIR, "dynamic_pricing.csv")
output_file = os.path.join(PROCESSED_DIR, "dynamic_pricing_processed_with_logical_regression.csv")

# Load the raw data file
df = pd.read_csv(input_file)

# Calculate demand-supply ratio and classify demand/supply levels
df["d_s_ratio"] = round(df["Number_of_Riders"] / df["Number_of_Drivers"], 2)
df["demand_class"] = np.where(df["Number_of_Riders"] > np.percentile(df["Number_of_Riders"], 75), "high_demand", "low_demand")
df["supply_class"] = np.where(df["Number_of_Drivers"] > np.percentile(df["Number_of_Drivers"], 75), "high_supply", "low_supply")

# Set up the target variable (binary classification: high demand vs. low demand)
y = np.where(df["demand_class"] == "high_demand", 1, 0)

# Select categorical features
categorical_features = ['Time_of_Booking', 'Vehicle_Type', 'demand_class', 'supply_class']

# Preprocess the categorical features with one-hot encoding
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ],
    remainder='passthrough'
)

# Create a pipeline with the preprocessor and logistic regression model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('scaler', StandardScaler(with_mean=False)),
    ('logreg', LogisticRegression())
])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[categorical_features], y, test_size=0.2, random_state=42)

# Train the logistic regression model
pipeline.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Logistic Regression Model Accuracy: {accuracy:.2f}")

# Save predictions in the original dataframe
df['Demand_Prediction'] = pipeline.predict(df[categorical_features])

# Save the processed dataset with predictions
df.to_csv(output_file, index=False)
print(f"Processed data with logistic regression predictions saved to {output_file}")
