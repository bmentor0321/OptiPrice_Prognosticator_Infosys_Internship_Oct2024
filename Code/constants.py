# constants.py

RAW_DATA_PATH = 'G:\\Spring board internship\\OptiPrice_Prognosticator_Infosys_Internship_Oct2024\\Data\\Raw\\dynamic_pricing.csv'
PROCESSED_DATA_PATH = 'G:\\Spring board internship\\OptiPrice_Prognosticator_Infosys_Internship_Oct2024\\Data\\Processed\\dynamic_pricing_results_with_pca.csv'
PROCESSED_DATA_DIR = 'G:\\Spring board internship\\OptiPrice_Prognosticator_Infosys_Internship_Oct2024\\Data\\Processed'

TARGET_COLUMN = 'Predicted'
NUMERIC_FEATURES = ['Number_of_Riders', 'Number_of_Drivers', 'Number_of_Past_Rides', 'Average_Ratings', 'Expected_Ride_Duration']
CATEGORICAL_FEATURES = ['Location_Category', 'Customer_Loyalty_Status', 'Time_of_Booking', 'Vehicle_Type']

TEST_SIZE = 0.2
RANDOM_STATE = 42
VAL_TEST_SPLIT = 0.5
