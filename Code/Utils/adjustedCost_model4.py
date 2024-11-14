import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np

# Paths
from constants import RAW_DIR, RESULTS_DIR

# Load the data
data = pd.read_csv(os.path.join(RAW_DIR, 'dynamic_pricing.csv'))

# Average rating class (good, bad) for average ratings
data["Average_Ratings_Class"] = np.where(data["Average_Ratings"] > 4.2, "Good", "Bad")

base_cost = data['Expected_Ride_Duration'].mean()
data.head()
