# Import necessary libraries
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# File paths
NEW_DATAPATH = r'D:\Rishi\Data\Raw\dynamic_pricing.csv'  
raw_data_path = r'D:\Rishi\Data\Raw\dynamic_pricing.csv' 
df = pd.read_csv(NEW_DATAPATH) 

# Ensure the processed data directory exists
processed_data_dir = r'D:\Rishi\Data\Processed' 

# Ensure the processed data directory exists
os.makedirs(processed_data_dir, exist_ok=True)

# Load the data
try:
    data = pd.read_csv(NEW_DATAPATH)
    print("Data loaded successfully.")
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit()

# Define a function to categorize ratings
def categorize_rating(rating):
    if rating >= 4.0:
        return 'Excellent'
    elif rating >= 3.0:
        return 'Good'
    else:
        return 'Poor'

# Add new column for rating categories
df['Rating_Category'] = df['Average_Ratings'].apply(categorize_rating)

# Remove 'Poor' ratings
df = df[df['Rating_Category'] != 'Poor']

# Combine specified categorical features into a single feature using one-hot encoding
combined_features = ['Customer_Loyalty_Status', 'Location_Category', 'Vehicle_Type', 'Time_of_Booking']
df_encoded = pd.get_dummies(df, columns=combined_features)

# Treat 'Historical_Cost_of_Ride' as a categorical variable by binning it
cost_bins = [0, 200, 400, np.inf]
cost_labels = ['Low', 'Medium', 'High']
df_encoded['Cost_Category'] = pd.cut(df_encoded['Historical_Cost_of_Ride'], bins=cost_bins, labels=cost_labels)

# One-hot encode the new cost category
cost_dummies = pd.get_dummies(df_encoded['Cost_Category'], prefix='Cost')
df_encoded = pd.concat([df_encoded, cost_dummies], axis=1)

# Drop the original 'Historical_Cost_of_Ride' and 'Cost_Category' columns
df_encoded.drop(['Historical_Cost_of_Ride', 'Cost_Category'], axis=1, inplace=True)

# Prepare features (X) and target (y)
features = [col for col in df_encoded.columns if col not in ['Average_Ratings', 'Rating_Category']]
X = df_encoded[features]
y = df_encoded['Rating_Category']

# Scale the numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train logistic regression model
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Print model performance
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Ensure the results directory exists
results_dir = r'D:\Rishi\Data\Results'
os.makedirs(results_dir, exist_ok=True)

# Define the output file path
output_file_path = os.path.join(results_dir, 'logistic_regression_model.csv')

# Write DataFrame to CSV at the specified path
df_encoded.to_csv(output_file_path, index=False)