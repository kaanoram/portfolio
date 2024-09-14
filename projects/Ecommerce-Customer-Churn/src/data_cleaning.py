import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Function to load the dataset
def load_data(file_path):
    """Loads the dataset from an Excel file."""
    return pd.read_excel(file_path, sheet_name='E Comm')

# Function to clean the dataset with improvements
def clean_data(df):
    """Handles missing values and outliers."""
    
    # Fill missing values for numerical columns with the median
    df['Tenure'] = df['Tenure'].fillna(df['Tenure'].median())
    df['WarehouseToHome'] = df['WarehouseToHome'].fillna(df['WarehouseToHome'].median())
    df['HourSpendOnApp'] = df['HourSpendOnApp'].fillna(df['HourSpendOnApp'].median())
    df['OrderAmountHikeFromlastYear'] = df['OrderAmountHikeFromlastYear'].fillna(df['OrderAmountHikeFromlastYear'].median())
    df['CouponUsed'] = df['CouponUsed'].fillna(df['CouponUsed'].median())
    df['OrderCount'] = df['OrderCount'].fillna(df['OrderCount'].median())
    df['DaySinceLastOrder'] = df['DaySinceLastOrder'].fillna(df['DaySinceLastOrder'].median())
    
    # Fill missing values for categorical columns with the mode
    df['PreferredLoginDevice'] = df['PreferredLoginDevice'].fillna(df['PreferredLoginDevice'].mode()[0])
    df['PreferredPaymentMode'] = df['PreferredPaymentMode'].fillna(df['PreferredPaymentMode'].mode()[0])
    df['PreferedOrderCat'] = df['PreferedOrderCat'].fillna(df['PreferedOrderCat'].mode()[0])
    
    # Removing outliers based on z-scores
    numeric_columns = ['Tenure', 'WarehouseToHome', 'HourSpendOnApp', 'OrderAmountHikeFromlastYear', 'OrderCount', 'DaySinceLastOrder']
    z_scores = np.abs((df[numeric_columns] - df[numeric_columns].mean()) / df[numeric_columns].std())
    df = df[(z_scores < 3).all(axis=1)]
    
    return df

# Function for encoding categorical variables
def encode_categorical(df):
    """Encodes categorical variables using LabelEncoder."""
    label_encoder = LabelEncoder()
    
    # List of categorical columns to encode
    categorical_cols = ['PreferredLoginDevice', 'PreferredPaymentMode', 'Gender', 'PreferedOrderCat', 'MaritalStatus']
    
    # Encoding each categorical column
    for col in categorical_cols:
        df[col] = label_encoder.fit_transform(df[col])
    
    return df

# Function for scaling the numerical features
def scale_data(df):
    """Scales the numerical features using StandardScaler."""
    scaler = StandardScaler()
    
    numeric_cols = ['Tenure', 'WarehouseToHome', 'HourSpendOnApp', 'OrderAmountHikeFromlastYear', 'OrderCount', 'DaySinceLastOrder']
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    return df
