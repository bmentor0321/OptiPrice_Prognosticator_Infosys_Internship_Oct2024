import os
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_DIR,"Data")
RAW_DIR = os.path.join(DATA_DIR,"Raw")
PROCESSED_DIR = os.path.join(DATA_DIR,"Processed")
RESULTS_DIR = os.path.join(DATA_DIR,"Results")
              
file_name = r'D:\Rishi\Data\Raw\dynamic_pricing.csv'
output_file_name = r'D:\Rishi\Data\Results'
processed_data_dir = r'D:\Rishi\Data\Processed' 
RAW_FILE_PATH = r'D:\Rishi\Data\Raw\dynamic_pricing.csv'
RESULT_FILES_DIR = r'D:\Rishi\Data\Results\result_files'
OUTPUT_FILES_DIR = r'D:\Rishi\Data\Results\output_files'

LOYALTY_DISCOUNTS = {"Regular": 0, "Silver": 0.05, "Gold": 0.10}
BASE_FARE = 50
PEAK_HOURS_MULTIPLIER = 1.5
TRAFFIC_MULTIPLIER = 1.2
PAST_RIDES_DISCOUNT = 0.01
DURATION_MULTIPLIER = 0.05
DURATION_THRESHOLD = 30
HISTORICAL_COST_ADJUSTMENT_FACTOR = 1.1
DISTANCE_THRESHOLD = 10