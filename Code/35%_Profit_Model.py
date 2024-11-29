import pandas as pd
df = pd.read_csv ('dynamic_pricing.csv')

""" DYNAMIC PRICING: IDEA TO GENERATE MORE PROFIT THAN THAT IN THE HISTORICAL COST OF RIDE

Idea: Increase the price when demand is higher than the supply. In the model, 20% price is increased when there is demand>supply.

Higher prices in urban areas (as they are more prime locations). In this model, 10% increased price in urban locations and 5% more in suburban areas and 2% increase in rural areas.

Discount for loyal customers (This would encourage more and more people to buy the gold pass, in long term, help the business to grow). In this model, gold pass gives 10% discount to the customer, silver one gives 5% discount while regular pass gives none.

Premium Vehicle type costs more than the economy one.(Requesting a premium vehicle provides premium servies and thus, price could be increased. In this model, 12% more charges for the premium vehicles """

def calculate_price(base_price, location, loyalty_status, vehicle_type, demand, supply):
    # Define multipliers
    location_multiplier = {'Urban': 1.10, 'Suburban': 1.05, 'Rural': 1.02}
    loyalty_discount = {'Gold': 0.9, 'Silver': 0.95, 'Regular': 1.0}
    vehicle_type_multiplier = {'Premium': 1.12, 'Economy': 1.0}
    
    # Adjust price based on location
    price = base_price * location_multiplier.get(location, 1.0)
    
    # Apply vehicle type multiplier
    price *= vehicle_type_multiplier.get(vehicle_type, 1.0)
    
    # Demand > Supply adjustment (increase price)
    if demand > supply:
        price *= 1.20  # 20% price increase due to demand-supply gap
    
    # Loyalty status discount
    price *= loyalty_discount.get(loyalty_status, 1.0)

    return price

# Apply the function to each row and create a new column 'Adjusted_Price'
df['Adjusted_Price'] = df.apply(lambda row: calculate_price(
    row['Historical_Cost_of_Ride'], 
    row['Location_Category'], 
    row['Customer_Loyalty_Status'], 
    row['Vehicle_Type'], 
    row['Number_of_Riders'], 
    row['Number_of_Drivers']
), axis=1)

# Display the DataFrame with the new 'Adjusted_Price' column
df.head()