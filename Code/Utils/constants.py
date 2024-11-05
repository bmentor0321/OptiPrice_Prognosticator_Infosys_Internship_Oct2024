import os


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_DIR,"Data")
RAW_DIR = os.path.join(DATA_DIR,"Raw")
PROCESSED_DIR = os.path.join(DATA_DIR,"Processed")
RESULTS_DIR = os.path.join(DATA_DIR,"Results")


#Constants

demand_supply_threshold = 2.3
constant_rate = 3.5  # Define the base rate per unit of duration, this is arround mean of ratio of ('Historical_Cost_of_Ride'/'Expected_Ride_Duration')
demand_hike = 0.35  # This is how much demand increase the pricing


