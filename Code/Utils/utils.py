import numpy as np
from Utils.constants import (
    HIGH_DEMAND_CUTOFF,
    LOW_DEMAND_CUTOFF,
    HIGH_SUPPLY_CUTOFF,
    LOW_SUPPLY_CUTOFF
)

def calculate_metrics(df):
    # Calculate demand-supply ratio and classify demand/supply levels
    df["d_s_ratio"] = round(df["Number_of_Riders"] / df["Number_of_Drivers"], 2)
    df["demand_class"] = np.where(df["Number_of_Riders"] > np.percentile(df["Number_of_Riders"], 75), "high_demand", "low_demand")
    df["supply_class"] = np.where(df["Number_of_Drivers"] > np.percentile(df["Number_of_Drivers"], 75), "high_supply", "low_supply")

    # Compute demand and supply metrics based on demand/supply classes
    df["demand_metric"] = np.where(df["demand_class"] == "high_demand",
                                   df["Number_of_Riders"] / np.percentile(df["Number_of_Riders"], 75),
                                   df["Number_of_Riders"] / np.percentile(df["Number_of_Riders"], 25))
    df["supply_metric"] = np.where(df["supply_class"] == "high_supply",
                                   df["Number_of_Drivers"] / np.percentile(df["Number_of_Drivers"], 75),
                                   df["Number_of_Drivers"] / np.percentile(df["Number_of_Drivers"], 25))
    return df

def adjust_price(row):
    # Adjust price based on demand-supply metrics
    if row['demand_metric'] > HIGH_DEMAND_CUTOFF and row['supply_metric'] < LOW_SUPPLY_CUTOFF:
        price = row['Historical_Cost_of_Ride'] * 1.10
    elif row['demand_metric'] < LOW_DEMAND_CUTOFF and row['supply_metric'] > HIGH_SUPPLY_CUTOFF:
        price = row['Historical_Cost_of_Ride'] * 0.95
    else:
        price = row['Historical_Cost_of_Ride']
    
    # Additional adjustments based on ride conditions
    if row['Average_Ratings'] >= 4.0:
        price *= 1.03
    if row['Time_of_Booking'] == 'Night':
        price *= 1.03
    if row['Vehicle_Type'] == 'Premium':
        price *= 1.03
    
    return price