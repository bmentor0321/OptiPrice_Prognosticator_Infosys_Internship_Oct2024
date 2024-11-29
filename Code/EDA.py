import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
df = pd.read_csv('dynamic_pricing.csv')

#To print the first few rows of dataset
print(df.head())

#To determine the shape/size of the dataset
print(df.shape)

#To determine the data type of each coloumn of the dataset
print(df.dtypes)

#To determine the number of unique values in each coloumn 
df.nunique()

#To print the coloumn heads
df.columns

#Distribution of location categories
sns.countplot(x='Location_Category', data=df)
plt.title('Distribution of Location Categories')
plt.show()

#Boxplot for number of past rides 
sns.boxplot(x='Number_of_Past_Rides', data=df)
plt.title('Number of Past Rides Boxplot')
plt.show()

#Expected ride duration
df['Expected_Ride_Duration'].hist(bins=20)
plt.title('Distribution of Expected Ride Duration')
plt.xlabel('Duration')
plt.ylabel('Frequency')
plt.show()

#Customer loyalty status distribution 
sns.countplot(x='Customer_Loyalty_Status', data=df)
plt.title('Customer Loyalty Status Distribution')
plt.show()

#To create a data dictionary
data_dict = pd.DataFrame(columns=['Column', 'Count', 'Unique Values', 'Range', 'Null Values', 'Possible values'])
for col in df.columns:
    count = df[col].shape[0]
    unique_values = df[col].nunique()
    
    if unique_values > 1:
        range_value = f"{df[col].min()} - {df[col].max()}"
    else:
        range_value = np.nan
    
    nulls = df[col].isnull().sum() 
    
    values = list(df[col].sample(frac=0.25, replace=False, random_state=42))
    values = list(set(values))[:5]  # Get unique values
    
    data_dict.loc[len(data_dict)] = [col, count, unique_values, range_value, nulls, values]
print(data_dict)

#Descriptive statistics of data
# Descriptive statistics for numerical columns
numerical_stats = df.describe()
print(numerical_stats)
# Descriptive statistics for categorical columns
categorical_stats = df.describe(include=['O'])
print(categorical_stats)

# Check data types of each column
data_types = df.dtypes
print(data_types)

# Convert 'Customer_Loyalty_Status' to category type
df['Customer_Loyalty_Status'] = df['Customer_Loyalty_Status'].astype('category')

#To check the new updated data types 
data_types = df.dtypes
print(data_types)

#Print the correlation matrix  
numerical_df = df.select_dtypes(include=[float,int])
corr_matrix = numerical_df.corr()
print (corr_matrix)

#Correlation plot
plt.figure (figsize=(7,7))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show

#No. of Drivers vs No. of Riders by Location Category
plt.figure(figsize=(4, 4))
sns.barplot(x='Location_Category', y='Number_of_Riders', data=df, estimator=sum, label='Number of Riders', color='b')
sns.barplot(x='Location_Category', y='Number_of_Drivers', data=df, estimator=sum, label='Number of Drivers', color='r')
plt.title('Number of Drivers vs Number of Riders by Location Category')
plt.ylabel('Count')
plt.legend()
plt.show()

# No. of Drivers vs No. of Riders by Vehicle Type
plt.figure(figsize=(4, 4))
sns.barplot(x='Vehicle_Type', y='Number_of_Riders', data=df, estimator=sum, label='Number of Riders', color='b')
sns.barplot(x='Vehicle_Type', y='Number_of_Drivers', data=df, estimator=sum, label='Number of Drivers', color='r')
plt.title('Number of Drivers vs Number of Riders by Vehicle Type')
plt.ylabel('Count')
plt.legend()
plt.show()

#Supply and Demand difference vs Historical cost
# Adding a new column 'Supply_Demand_Difference'
df['Supply_Demand_Difference'] = df['Number_of_Riders'] - df['Number_of_Drivers']
plt.figure(figsize=(10,9))
plt.scatter(df['Supply_Demand_Difference'], df['Historical_Cost_of_Ride'], alpha=0.6, color='blue')
plt.title('Supply and Demand Difference vs Historical Cost')
plt.xlabel('Supply and Demand Difference')
plt.ylabel('Historical Cost of Ride')
plt.grid()
plt.show()

# Plotting Historical Cost vs Vehicle Type
plt.figure(figsize=(12, 6))
sns.boxplot(x='Vehicle_Type', y='Historical_Cost_of_Ride', data=df)
plt.title('Historical Cost vs Vehicle Type')
plt.xlabel('Vehicle Type')
plt.ylabel('Historical Cost of Ride')
plt.grid()
plt.show()