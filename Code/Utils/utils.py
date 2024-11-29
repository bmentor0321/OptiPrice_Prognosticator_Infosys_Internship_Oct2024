from Utils.constants import demand_hike
import numpy as np

# Functions to calculate surge charges based on Demand_supply class
def apply_surge(df):
    SD_surge_charge=0
    if (df['Demand_class']=='Higher_demand') & (df['Supply_class']=='Lower_supply'):
        SD_surge_charge = df['base_cost'] * (demand_hike * df['demand_supply_factor'])
    return SD_surge_charge


# Function to calculate vehicle-specific charges
def Vehicle_charge(df):
    if df['Vehicle_Type'] == 'Premium':
        return df['base_cost'] * 0.05
    else:  # Economy
        return df['base_cost'] * 0.025

# Function to calculate location and time-based surge charges
def Cal_loyalty_discount(df):
    if df['Customer_Loyalty_Status'] == "Gold":
        loyalty_discount = df['base_cost'] * 0.10  # 10% discount
    elif df['Customer_Loyalty_Status'] == "Silver":
        loyalty_discount = df['base_cost'] * 0.05  # 5% discount
    else:  # Regular
        loyalty_discount = df['base_cost'] * 0.01  # 1% discount
    return loyalty_discount

#Function to itentify the peak_hours
def peak_demand(df):
    
    # Group by Location Category and Time of Booking and calculate the mean supply-demand ratio for each group
    mean_ratios = df.groupby(['Location_Category', 'Time_of_Booking'])['Demand_Supply_Ratio'].transform('mean')
    
    df['Peak_hour'] = np.where(df['Demand_Supply_Ratio']>mean_ratios,"Peak hours","low peak")
    
    return df['Peak_hour']

def Location_time_surge(df):
    if df['Peak_hour'] == "Peak hours":
            if df['Location_Category'] == 'Urban':
                return df['base_cost'] * 0.10
            elif df['Location_Category'] == 'Suburban':
                return df['base_cost'] * 0.05
            elif df['Location_Category'] == 'Rural':
                return df['base_cost'] * 0.03
    return 0

