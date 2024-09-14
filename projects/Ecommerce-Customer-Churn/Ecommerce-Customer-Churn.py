import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, classification_report


# Function to load the dataset
def load_data(file_path):
    """Loads the dataset from an Excel file."""
    return pd.read_excel(file_path, sheet_name='E Comm')

# Function to clean the dataset
def clean_data(df):
    """Handles missing values by filling with median values."""
    df['Tenure'] = df['Tenure'].fillna(df['Tenure'].median())
    df['WarehouseToHome'] = df['WarehouseToHome'].fillna(df['WarehouseToHome'].median())
    df['HourSpendOnApp'] = df['HourSpendOnApp'].fillna(df['HourSpendOnApp'].median())
    df['OrderAmountHikeFromlastYear'] = df['OrderAmountHikeFromlastYear'].fillna(df['OrderAmountHikeFromlastYear'].median())
    df['CouponUsed'] = df['CouponUsed'].fillna(df['CouponUsed'].median())
    df['OrderCount'] = df['OrderCount'].fillna(df['OrderCount'].median())
    df['DaySinceLastOrder'] = df['DaySinceLastOrder'].fillna(df['DaySinceLastOrder'].median())
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

# Function to preprocess data for model
def preprocess_data(df):
    """Prepares the data for modeling by encoding and splitting."""
    
    # Encoding categorical features
    df = encode_categorical(df)
    
    # Separating features (X) and target variable (y)
    X = df.drop(['CustomerID', 'Churn'], axis=1)
    y = df['Churn']
    
    # Splitting the data into training and test sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    return X_train, X_test, y_train, y_test

# Function to train and evaluate a Logistic Regression model
def train_logistic_regression(X_train, X_test, y_train, y_test):
    """Trains a Logistic Regression model and evaluates it."""
    
    # Initializing the Logistic Regression model
    model = LogisticRegression(max_iter=1000)
    
    # Fitting the model on the training data
    model.fit(X_train, y_train)
    
    # Making predictions on the test set
    y_pred = model.predict(X_test)
    
    # Evaluating the model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    
    # Displaying the classification report
    report = classification_report(y_test, y_pred)
    
    print("Model Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"AUC-ROC: {roc_auc:.4f}")
    print("\nClassification Report:")
    print(report)

# Main function to run the model training and evaluation process
def run_model_pipeline(file_path):
    """Runs the complete pipeline for model building and evaluation."""
    
    # Step 1: Load and clean the dataset
    df = load_data(file_path)
    df = clean_data(df)
    
    # Step 2: Preprocess the data
    X_train, X_test, y_train, y_test = preprocess_data(df)
    
    # Step 3: Train and evaluate a Logistic Regression model
    train_logistic_regression(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    # Path to the dataset
    file_path = 'data/E-Commerce-Dataset.xlsx'
    # Run the churn analysis workflow
    run_churn_analysis(file_path)
    # Run the model pipeline
    run_model_pipeline(file_path)