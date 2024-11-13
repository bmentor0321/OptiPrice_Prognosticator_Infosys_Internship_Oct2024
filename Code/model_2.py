{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "e467b870-1758-4625-b3e3-89c01ab445e0",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed data saved to output\\model2_errors.csv\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "from sklearn.linear_model import LinearRegression, Lasso, Ridge\n",
    "#from Utils.constants import RAW_DIR, PROCESSED_DIR\n",
    "import os\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.svm import SVR\n",
    "from sklearn.metrics import mean_squared_error\n",
    "\n",
    "\n",
    "\n",
    "# Load the dataset\n",
    "data = pd.read_csv(\"dynamic_pricing.csv\")\n",
    "\n",
    "\n",
    "# Select features and target variable\n",
    "X = data[['Number_of_Riders', 'Number_of_Drivers', 'Number_of_Past_Rides', 'Average_Ratings', 'Expected_Ride_Duration']]\n",
    "y = data['Historical_Cost_of_Ride']\n",
    "\n",
    "# Split the data into train, validation, and test sets\n",
    "X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)\n",
    "X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)\n",
    "\n",
    "# Standardize the features\n",
    "scaler = StandardScaler()\n",
    "X_train = scaler.fit_transform(X_train)\n",
    "X_val = scaler.transform(X_val)\n",
    "X_test = scaler.transform(X_test)\n",
    "\n",
    "# Initialize models\n",
    "models = {\n",
    "    \"Linear Regression\": LinearRegression(),\n",
    "    \"Lasso\": Lasso(),\n",
    "    \"Ridge\": Ridge(),\n",
    "    \"SVR\": SVR()\n",
    "}\n",
    "\n",
    "# Dictionary to store errors\n",
    "results = {\n",
    "    \"Model\": [],\n",
    "    \"Train Error\": [],\n",
    "    \"Validation Error\": [],\n",
    "    \"Test Error\": []\n",
    "}\n",
    "\n",
    "# Train each model and calculate errors\n",
    "for name, model in models.items():\n",
    "    # Fit the model on the training data\n",
    "    model.fit(X_train, y_train)\n",
    "\n",
    "    # Predict on train, validation, and test sets\n",
    "    y_train_pred = model.predict(X_train)\n",
    "    y_val_pred = model.predict(X_val)\n",
    "    y_test_pred = model.predict(X_test)\n",
    "\n",
    "    # Calculate RMSE for train, validation, and test sets\n",
    "    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))\n",
    "    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))\n",
    "    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))\n",
    "\n",
    "    # Store the results\n",
    "    results[\"Model\"].append(name)\n",
    "    results[\"Train Error\"].append(train_rmse)\n",
    "    results[\"Validation Error\"].append(val_rmse)\n",
    "    results[\"Test Error\"].append(test_rmse)\n",
    "\n",
    "# Convert results to a DataFrame\n",
    "results_data = pd.DataFrame(results)\n",
    "\n",
    "# Save the results to a CSV file\n",
    "# output_file = os.path.join(PROCESSED_DIR, \"model2_errors.csv\")\n",
    "# data.to_csv(output_file, index=False)\n",
    "import os\n",
    "\n",
    "PROCESSED_DIR = \"output\"  # Replace with your actual directory path\n",
    "output_file = os.path.join(PROCESSED_DIR, \"model2_errors.csv\")\n",
    "data.to_csv(output_file, index=False)\n",
    "\n",
    "\n",
    "print(f\"Processed data saved to {output_file}\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d5e04406-c1ce-4104-8a75-307d208b0f07",
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