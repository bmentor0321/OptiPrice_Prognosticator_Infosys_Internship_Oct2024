# Import necessary libraries
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from Utils.constants import RAW_DIR, file_name, RESULTS_DIR, output_file_name, processed_data_dir
from Utils.utils import get_mean

# File paths
NEW_DATAPATH = r'D:\Rishi\Data\Raw\dynamic_pricing.csv'  
df = data = RAW_DIR = pd.read_csv(os.path.join(RAW_DIR, file_name))

# Load the data
try:
    df = pd.read_csv(NEW_DATAPATH)
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

# Split the data into train, validation, and test sets (60% train, 20% validation, 20% test)
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Train logistic regression model on the training set
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)

# Calculate accuracy and error rates
train_accuracy = model.score(X_train, y_train)
validation_accuracy = model.score(X_val, y_val)
test_accuracy = model.score(X_test, y_test)

train_error = 1 - train_accuracy
validation_error = 1 - validation_accuracy
test_error = 1 - test_accuracy

# Print model performance and error rates
print("Train Error:", train_error)
print("Validation Error:", validation_error)
print("Test Error:", test_error)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Ensure the results directory exists
RESULTS_DIR = r'D:\Rishi\Data\Results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# Define the output file path
output_file_path = os.path.join(RESULTS_DIR, 'model_3_result.csv')

# Write DataFrame to CSV at the specified path
df_encoded.to_csv(output_file_path, index=False)