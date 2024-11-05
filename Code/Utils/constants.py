import os

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_DIR,"Data")
RAW_DIR = os.path.join(DATA_DIR,"Raw")
PROCESSED_DIR = os.path.join(DATA_DIR,"Processed")
RESULTS_DIR = os.path.join(DATA_DIR,"Results")
              

file_name = r'D:\Rishi\Data\Raw\dynamic_pricing.csv'
output_file_name = r'D:\Rishi\Data\Results'
processed_data_dir = r'D:\Rishi\Data\Processed' 

train_split = 0.6
val_split = 0.2
test_split = 0.2

