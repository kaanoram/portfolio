import pandas as pd
import matplotlib.pyplot as plt

# Function to load the dataset
def load_data(file_path):
    """Loads the dataset from an Excel file."""
    return pd.read_excel(file_path, sheet_name='E Comm')

# Function to clean the dataset
def clean_data(df):
    """Handles missing values by filling with median values."""
    df['Tenure'].fillna(df['Tenure'].median(), inplace=True)
    df['WarehouseToHome'].fillna(df['WarehouseToHome'].median(), inplace=True)
    df['HourSpendOnApp'].fillna(df['HourSpendOnApp'].median(), inplace=True)
    df['OrderAmountHikeFromlastYear'].fillna(df['OrderAmountHikeFromlastYear'].median(), inplace=True)
    df['CouponUsed'].fillna(df['CouponUsed'].median(), inplace=True)
    df['OrderCount'].fillna(df['OrderCount'].median(), inplace=True)
    df['DaySinceLastOrder'].fillna(df['DaySinceLastOrder'].median(), inplace=True)
    return df

# Function to plot churn distribution
def plot_churn_distribution(df):
    """Plots the distribution of churn in the dataset."""
    plt.figure(figsize=(6,4))
    df['Churn'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
    plt.title('Churn Distribution')
    plt.xlabel('Churn')
    plt.ylabel('Number of Customers')
    plt.xticks(ticks=[0, 1], labels=['No Churn', 'Churn'], rotation=0)
    plt.show()

# Function to plot average values for churned vs non-churned customers
def plot_churn_vs_features(df, feature, title, xlabel, ylabel, color):
    """Plots the relationship between churn and a specific feature."""
    plt.figure(figsize=(6,4))
    df.groupby('Churn')[feature].mean().plot(kind='bar', color=color)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(ticks=[0, 1], labels=['No Churn', 'Churn'], rotation=0)
    plt.show()

# Main function to run the analysis
def run_churn_analysis(file_path):
    """Executes the churn analysis workflow."""
    
    # Step 1: Load the dataset
    df = load_data(file_path)
    
    # Step 2: Clean the dataset
    df = clean_data(df)
    
    # Step 3: Plot churn distribution
    plot_churn_distribution(df)
    
    # Step 4: Explore relationships between churn and key features
    plot_churn_vs_features(df, 'Tenure', 'Average Tenure by Churn', 'Churn', 'Average Tenure', 'lightgreen')
    plot_churn_vs_features(df, 'SatisfactionScore', 'Average Satisfaction Score by Churn', 'Churn', 'Average Satisfaction Score', 'lightcoral')
    plot_churn_vs_features(df, 'CouponUsed', 'Average Coupons Used by Churn', 'Churn', 'Average Coupons Used', 'lightblue')
    plot_churn_vs_features(df, 'OrderCount', 'Average Order Count by Churn', 'Churn', 'Average Order Count', 'lightgreen')

# Path to the dataset
file_path = '/mnt/data/E Commerce Dataset.xlsx'

# Run the churn analysis workflow
run_churn_analysis(file_path)
