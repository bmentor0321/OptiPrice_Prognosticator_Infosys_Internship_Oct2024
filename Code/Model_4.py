import numpy as np
import pandas as pd

# Load the dataset
df = pd.read_csv('dynamic_pricing.csv')

# Calculate Revenue
df['Revenue'] = df['Number_of_Riders'] * df['Historical_Cost_of_Ride']

# Calculate Total Cost (assuming Historical Cost is the cost)
df['Total_Cost'] = df['Historical_Cost_of_Ride']  # or another cost calculation if applicable

# Calculate Profit
df['Total_Profit'] = df['Revenue'] - df['Total_Cost']

# Calculate Revenue_Per_Minute
df['Revenue_Per_Minute'] = df['Revenue'] / df['Expected_Ride_Duration']

# Calculate Profit_Per_Minute
df['Profit_Per_Minute'] = df['Total_Profit'] / df['Expected_Ride_Duration']

# Calculate Cost_Per_Transaction
df['Cost_Per_Transaction'] = df['Total_Cost'] / df['Number_of_Riders']

# Calculate Revenue_Per_Transaction
df['Revenue_Per_Transaction'] = df['Revenue'] / df['Number_of_Riders']

# Select relevant columns to save
output_df = df[['Location_Category','Number_of_Drivers','Number_of_Riders', 'Revenue', 'Total_Cost', 'Total_Profit', 'Revenue_Per_Minute', 'Profit_Per_Minute','Cost_Per_Transaction','Revenue_Per_Transaction']]

# Save to CSV
output_df.to_csv('ride_metrics_output.csv', index=False)

print("Output saved to 'model_4_output.csv'")
