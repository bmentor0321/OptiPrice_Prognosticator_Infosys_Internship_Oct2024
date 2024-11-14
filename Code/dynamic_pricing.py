import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
import numpy as np
import constants as C  # Import constants from the constants.py file

# Load dataset using the path from constants.py
data = pd.read_csv(C.RAW_DATA_PATH)

# First feature: Supply-Demand Ratio
data['Supply_Demand_Ratio'] = data['Number_of_Drivers'] / data['Number_of_Riders']

# Second feature: Customer Value Score
normalized_rides = data['Number_of_Past_Rides'] / data['Number_of_Past_Rides'].max()
normalized_ratings = (data['Average_Ratings'] - data['Average_Ratings'].min()) / \
                    (data['Average_Ratings'].max() - data['Average_Ratings'].min())

loyalty_weights = {
    'Regular': 0.5,
    'Silver': 0.75,
    'Gold': 1.0,
    'Platinum': 1.25
}
loyalty_score = data['Customer_Loyalty_Status'].map(loyalty_weights)

data['Customer_Value_Score'] = (normalized_rides * 0.4 + 
                               normalized_ratings * 0.3 + 
                               loyalty_score * 0.3)

# Third feature: Time_Based_Demand_Multiplier
time_demand_weights = {
    'Early_Morning': 1.2,
    'Morning': 1.5,
    'Afternoon': 1.0,
    'Evening': 1.4,
    'Night': 1.3,
    'Late_Night': 1.6
}

location_weights = {
    'Urban': 1.3,
    'Suburban': 1.1,
    'Rural': 1.0
}

# Calculate Time Based Demand Multiplier
time_multiplier = data['Time_of_Booking'].map(time_demand_weights)
location_multiplier = data['Location_Category'].map(location_weights)
data['Time_Based_Demand_Multiplier'] = time_multiplier * location_multiplier


# Feature and target selection using constants from constants.py
X = data[C.NUMERIC_FEATURES + C.CATEGORICAL_FEATURES]
Y = data[C.TARGET_COLUMN]

# Preprocessing with ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), C.NUMERIC_FEATURES),
        ('cat', OneHotEncoder(), C.CATEGORICAL_FEATURES)
    ]
)

# Pipeline setup: Preprocessing + PCA + Model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('pca', PCA(n_components=0.95)),
    ('model', LinearRegression())
])

# Data splitting using constants from constants.py
X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=C.TEST_SIZE, random_state=C.RANDOM_STATE)
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=C.VAL_TEST_SPLIT, random_state=C.RANDOM_STATE)

# Train the model
pipeline.fit(X_train, Y_train)

# Error calculations function
def calculate_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Calculate train, validation, and test errors
train_error = calculate_error(Y_train, pipeline.predict(X_train))
val_error = calculate_error(Y_val, pipeline.predict(X_val))
test_error = calculate_error(Y_test, pipeline.predict(X_test))

# Store results in a DataFrame
results = pd.DataFrame({
    'Model': ['Model_1 (Linear Regression)'],
    'Model_Description': ['Using numeric and categorical columns with PCA'],
    'Train_Error': [train_error],
    'Val_Error': [val_error],
    'Test_Error': [test_error]
})

# Displaying the results in a clean format
print("\nModel Performance Summary:\n")
print(results.to_string(index=False))

# Save results using the processed data path from constants.py
os.makedirs(C.PROCESSED_DATA_DIR, exist_ok=True)
results.to_csv(os.path.join(C.PROCESSED_DATA_DIR, 'model_errors.csv'), index=False)

# Print results to the console
print(results)
