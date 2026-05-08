'''
Example Code: Data Handling and Preprocessing
This code will showcase:

*Loading the dataset
*Handling missing values
*Normalizing numerical columns
*Encoding categorical variables
'''


# Import required libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv("AB_NYC_2019.csv")  # Change path if necessary

# 1. Overview of the Dataset
print("Dataset Overview:")
print(data.info())
print("\nSample Data:")
print(data.head())

# 2. Handling Missing Values
# Checking for missing values
missing_values = data.isnull().sum()
print("\nMissing Values per Column:")
print(missing_values)

# Dropping columns with high missing values (e.g., 'last_review' and 'reviews_per_month')
data = data.drop(columns=['last_review', 'reviews_per_month'], axis=1)

# Filling missing values in 'name' and 'host_name' columns with placeholder
data['name'] = data['name'].fillna('Unknown')
data['host_name'] = data['host_name'].fillna('Unknown')

# 3. Normalizing Data
# Selecting numerical columns for normalization
num_cols = ['price', 'minimum_nights', 'number_of_reviews']

# Using StandardScaler to normalize data
scaler = StandardScaler()
data[num_cols] = scaler.fit_transform(data[num_cols])

print("\nNormalized Data Sample:")
print(data[num_cols].head())

# 4. Encoding Categorical Variables
# Using one-hot encoding for 'neighbourhood_group' and 'room_type'
data = pd.get_dummies(data, columns=['neighbourhood_group', 'room_type'], drop_first=True)

print("\nData with Encoded Categorical Variables:")
print(data.head())

# Final overview of processed data
print("\nProcessed Data Overview:")
print(data.info())
