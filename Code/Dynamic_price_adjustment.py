# Importing Libraries
import pandas as pd
import numpy as np
import os
from Utils.constants import RAW_DIR, PROCESSED_DIR
from Utils.constants import demand_supply_threshold,constant_rate
from Utils.utils import apply_surge, Cal_loyalty_discount,peak_demand,Vehicle_charge, Location_time_surge

# Load the dataset
input_file = os.path.join(RAW_DIR, "dynamic_pricing.csv")
df= pd.read_csv(input_file)


# Calculating Demand_Supply_Ratio and picking demand_supply_threshold = 2.5 arround the mean of Demand_Supply_Ratio
# 1. Higher Demand = when 'Demand_Supply_Ratio' > demand_supply_threshold (2.5) else Low-demand
# 2. Higher supply = when 'Demand_Supply_Ratio' < demand_supply_threshold (2.5) else Low-supply

df['Demand_Supply_Ratio'] = df['Number_of_Riders'] / df['Number_of_Drivers']

df['Demand_class'] = np.where(df['Demand_Supply_Ratio'] > demand_supply_threshold, "Higher_demand", "Lower_demand")
df['Supply_class'] = np.where(df['Demand_Supply_Ratio'] < demand_supply_threshold, "Higher_supply", "Lower_supply")


# calulation Base Price and Surge_charge based on supply demand ratio and demand_supply_factor
# 1. Calculate base historical cost based on expected_Ride_duration
# 2. Calculate rider-to-driver ratio
# 3. Calculate demand-supply factor
# 4. Defining a methode to Calculate supply_demand_surge and Apply the dynamic pricing formula


# Calculate base historical cost based on expected_Ride_duration
df['base_cost'] = df['Expected_Ride_Duration'] * constant_rate

# Calculate demand-supply factor
df['demand_supply_factor'] = df['Demand_Supply_Ratio'] - 1
df['demand_supply_factor'] = df['demand_supply_factor'].apply(lambda x: min(x, 7))

peak_demand(df)
# defining a methode to Calculate supply_demand_surge and Apply the dynamic pricing formula
df['S/D_surge_charge'] = df.apply(apply_surge,axis=1)

# Conditional Surge based on Vehical_Type and Time_of_booking && Location_Category Condition
df['Vehical_charge'] = df.apply(Vehicle_charge, axis=1)
df['Location_time_surge'] = df.apply(Location_time_surge, axis=1)
df['loyalty_discount'] = df.apply(Cal_loyalty_discount, axis=1)


# Calculating New_adjusted_cost

df['New_cost']= df['base_cost'] + df['S/D_surge_charge'] + df['Vehical_charge']+ df['Location_time_surge'] - df['loyalty_discount']

# Revenue Before and after
print("Revenue before applying Dynamic_pricing -->",round(sum(df['Historical_Cost_of_Ride']),2))
print("Revenue after applying Dynamic_pricing-->",round(sum(df['New_cost']),2))

diff=sum(df['New_cost'])-sum(df['Historical_Cost_of_Ride'])
print("Diffrenece of Revenue--> ", diff)
print("Revenue Percentage --> ", diff/sum(df['Historical_Cost_of_Ride'])*100)
output_file = os.path.join(PROCESSED_DIR, "Dynamic_pricing adjustment with new cost.csv")
df.to_csv(output_file, index=False)

# Conclusion
# Diffrenece of Revenue-->  179795.39
# Revenue Percentage -->  48.26
