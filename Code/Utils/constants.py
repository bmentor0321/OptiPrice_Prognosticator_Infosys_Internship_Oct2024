import os

# Directory paths
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(_file_))))
DATA_DIR = os.path.join(PROJECT_DIR, "Data")
RAW_DIR = os.path.join(DATA_DIR, "Raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "Processed")
RESULTS_DIR = os.path.join(DATA_DIR, "Results")

# Price adjustment constants for the demand-supply metrics i kept in the utis file
HIGH_DEMAND_CUTOFF = 1.15
LOW_DEMAND_CUTOFF = 0.85
HIGH_SUPPLY_CUTOFF = 1.15
LOW_SUPPLY_CUTOFF = 0.85
              