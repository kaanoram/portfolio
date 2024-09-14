import pandas as pd
from data_cleaning import load_data, clean_data
from clustering import customer_segmentation, plot_customer_segments, find_optimal_clusters
from models import preprocess_data, train_multiple_models, train_nn_model
from eda import eda

# Main function to run the full process with additional models
def run_analysis_and_models(file_path):
    """Runs the full process with data cleaning, EDA, scaling, and model training."""
    
    # Step 1: Load the dataset
    df = load_data(file_path)
    
    # Step 2: Clean the dataset
    df = clean_data(df)
    
        # Step 3: Perform EDA (Exploratory Data Analysis)
    print("Performing Exploratory Data Analysis (EDA)...")
    eda(df)  # Reintroduced EDA here for visualization and understanding
    
    # Step 4: Customer Segmentation (Clustering)
    
    # Optional: Find the optimal number of clusters using the elbow method
    print("Finding the optimal number of clusters...")
    optimal_clusters = find_optimal_clusters(df, max_clusters=10)  # This will plot the elbow method and return optimal clusters
    
    # Perform customer segmentation using KMeans
    print(f"Performing customer segmentation with {optimal_clusters} clusters...")
    segmented_df, kmeans_model = customer_segmentation(df, num_clusters=optimal_clusters)
    
    # Visualize customer segmentation
    plot_customer_segments(segmented_df)
    
    # Step 5: Machine Learning Models (Logistic Regression, Random Forest, Gradient Boosting, Neural Network)
    
    # Preprocess the data for modeling (encode, scale, and split)
    print("Preprocessing data for model training...")
    X_train, X_test, y_train, y_test = preprocess_data(df)  # Scaling happens in preprocess_data
    
    # Train and evaluate traditional models (Logistic Regression, Random Forest, Gradient Boosting)
    print("Training traditional machine learning models...")
    performance = train_multiple_models(X_train, X_test, y_train, y_test)
    performance_df = pd.DataFrame(performance).T
    
    # Train and evaluate Neural Network model
    print("Training Neural Network model...")
    nn_performance = train_nn_model(X_train, X_test, y_train, y_test)
    nn_performance_df = pd.DataFrame(nn_performance, index=['Neural Network'])
    
    # Combine traditional models and NN performance into a single dataframe
    performance_df = pd.concat([performance_df, nn_performance_df])
    
    # Step 6: Output performance results
    print("Model Performance Comparison:")
    print(performance_df)
    
    return performance_df

# Example usage
if __name__ == "__main__":
    file_path = '../data/E-Commerce-Dataset.xlsx'  # Replace with your dataset path
    performance_df = run_analysis_and_models(file_path)
    print(performance_df)
