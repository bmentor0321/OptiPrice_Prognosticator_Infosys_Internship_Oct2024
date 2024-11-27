import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from Utils.utils import get_mean
from Utils.constants import RAW_DIR ,RESULT_FILES_DIR, RAW_FILE_PATH, OUTPUT_FILES_DIR, LOYALTY_DISCOUNTS, BASE_FARE,PEAK_HOURS_MULTIPLIER, TRAFFIC_MULTIPLIER, PAST_RIDES_DISCOUNT, DURATION_MULTIPLIER, DURATION_THRESHOLD, HISTORICAL_COST_ADJUSTMENT_FACTOR, DISTANCE_THRESHOLD,output_file_name
 
# Load the data
try:
    df = pd.read_csv(RAW_FILE_PATH)
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

# Label encode the target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Scale the numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into train, validation, and test sets (60% train, 20% validation, 20% test)
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y_encoded, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Train logistic regression model on the training set (multiclass classification)
model = LogisticRegression(random_state=42, max_iter=1000, multi_class='ovr')
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Ensure the results directory exists for model output
os.makedirs(OUTPUT_FILES_DIR, exist_ok=True)

# Calculate ERROR and ACCURACY rates
train_error = model.score(X_train, y_train)
validation_error = model.score(X_val, y_val)
test_error = model.score(X_test, y_test)

# Calculate ACCURACY rates as (1 - ERROR)
train_accuracy = 1 - train_error
validation_accuracy = 1 - validation_error
test_accuracy = 1 - test_error

# Print model performance (as error rates)
print("Train Error:", train_error * 100)  # Convert to percentage
print("Validation Error:", validation_error * 100)  # Convert to percentage
print("Test Error:", test_error * 100)  # Convert to percentage

# Print model performance and error rates (reversed)
print("Train Accuracy (as error):", train_accuracy * 100)  # Convert to percentage
print("Validation Accuracy (as error):", validation_accuracy * 100)  # Convert to percentage
print("Test Accuracy (as error):", test_accuracy * 100)  # Convert to percentage

# Make predictions on the test set
y_pred = model.predict(X_test)

# Ensure the directory exists
os.makedirs(OUTPUT_FILES_DIR, exist_ok=True)
output_file_path = os.path.join(OUTPUT_FILES_DIR, 'model_3_result.csv')
df_encoded.to_csv(output_file_path, index=False)
comparison_file_path = os.path.join(RESULT_FILES_DIR, "comparison.csv")

# Prepare the metrics for comparison.csv
df_metrics = pd.DataFrame(columns=['Models', "Model Description", "Train error", "Val error", "Test error"])
# Add the current model's metrics to the DataFrame (use error rates)
df_metrics.loc[len(df_metrics)] = ["Model 3", "All Numerical Features + Standard Scaler Introduced + Categorical Features One Hot Encoded",train_error * 100, validation_error * 100, test_error * 100 ]

# Check if comparison.csv already exists in the specified path, then append; otherwise, create a new file
if os.path.exists(comparison_file_path):
    # Read the existing metrics file
    df_metrics_ = pd.read_csv(comparison_file_path)
    
    # Concatenate the new metrics to the existing ones
    df_metrics_ = pd.concat([df_metrics_, df_metrics], ignore_index=True)
else:
    # If the file doesn't exist, start with the current metrics
    df_metrics_ = df_metrics

# Save the updated metrics back to comparison.csv
df_metrics_.to_csv(comparison_file_path, index=False)