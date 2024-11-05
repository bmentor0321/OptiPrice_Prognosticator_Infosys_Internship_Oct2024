from Utils.constants import demand_hike

#Functions
def apply_surge(df):
    SD_surge_charge=0
    if (df['Demand_class']=='Higher_demand') & (df['Supply_class']=='Lower_supply'):
        SD_surge_charge = df['base_cost'] * (demand_hike * df['demand_supply_factor'])
    return SD_surge_charge

def cal_surge_charge(df):
    surge_charge = 0
    if df['Vehicle_Type'] == 'Premium':
        if (df['Location_Category'] in ('Urban', 'Suburban')) & (df['Time_of_Booking'] in ('Evening', 'Night')):
            surge_charge = df['base_cost'] * 0.05 + df['base_cost'] * 0.02
    else:
        if (df['Location_Category'] in ('Urban', 'Suburban')) & (df['Time_of_Booking'] in ('Evening', 'Night')):
            surge_charge = df['base_cost'] * 0.025 + df['base_cost'] * 0.01
    return surge_charge