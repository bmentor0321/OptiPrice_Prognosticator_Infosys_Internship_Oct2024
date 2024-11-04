import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from constants import DATASET_PATH, FEATURES_TO_SCALE, TARGET_VARIABLE

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)

    # Selecting numerical features only
    X = df.select_dtypes(include=['float64', 'int64']).drop(columns=[TARGET_VARIABLE])
    y = df[TARGET_VARIABLE]

    # Splitting the dataset
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Feature scaling for selected columns
    scaler = StandardScaler()
    numerical_features = FEATURES_TO_SCALE
    X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
    X_val[numerical_features] = scaler.transform(X_val[numerical_features])
    X_test[numerical_features] = scaler.transform(X_test[numerical_features])



    return X_train, X_val, X_test, y_train, y_val, y_test
