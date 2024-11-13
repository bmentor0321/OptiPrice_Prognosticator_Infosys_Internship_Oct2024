{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "4f262e94-f74d-485a-9290-44ea97555205",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Output saved to 'ride_metrics_output.csv'\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "\n",
    "# Load the dataset\n",
    "df = pd.read_csv('dynamic_pricing.csv')\n",
    "\n",
    "# Calculate Revenue\n",
    "df['Revenue'] = df['Number_of_Riders'] * df['Historical_Cost_of_Ride']\n",
    "\n",
    "# Calculate Total Cost (assuming Historical Cost is the cost)\n",
    "df['Total_Cost'] = df['Historical_Cost_of_Ride']  # or another cost calculation if applicable\n",
    "\n",
    "# Calculate Profit\n",
    "df['Total_Profit'] = df['Revenue'] - df['Total_Cost']\n",
    "\n",
    "# Calculate Revenue_Per_Minute\n",
    "df['Revenue_Per_Minute'] = df['Revenue'] / df['Expected_Ride_Duration']\n",
    "\n",
    "# Calculate Profit_Per_Minute\n",
    "df['Profit_Per_Minute'] = df['Total_Profit'] / df['Expected_Ride_Duration']\n",
    "\n",
    "# Calculate Cost_Per_Transaction\n",
    "df['Cost_Per_Transaction'] = df['Total_Cost'] / df['Number_of_Riders']\n",
    "\n",
    "# Calculate Revenue_Per_Transaction\n",
    "df['Revenue_Per_Transaction'] = df['Revenue'] / df['Number_of_Riders']\n",
    "\n",
    "# Select relevant columns to save\n",
    "output_df = df[['Location_Category','Number_of_Drivers','Number_of_Riders', 'Revenue', 'Total_Cost', 'Total_Profit', 'Revenue_Per_Minute', 'Profit_Per_Minute','Cost_Per_Transaction','Revenue_Per_Transaction']]\n",
    "\n",
    "# Save to CSV\n",
    "output_df.to_csv('ride_metrics_output.csv', index=False)\n",
    "\n",
    "print(\"Output saved to 'ride_metrics_output.csv'\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cd4a0de9-3dee-4501-85ba-ad3990e8e8cb",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
