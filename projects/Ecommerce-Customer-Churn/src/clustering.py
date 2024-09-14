from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd

# Function to perform customer segmentation
def customer_segmentation(df, num_clusters):
    """Performs KMeans clustering for customer segmentation."""
    
    # Select features relevant for customer segmentation
    features = ['OrderCount', 'OrderAmountHikeFromlastYear', 'HourSpendOnApp', 'SatisfactionScore']
    customer_data = df[features].dropna()  # Drop rows with missing values in these columns
    
    # Scale the data
    scaler = StandardScaler()
    customer_data_scaled = scaler.fit_transform(customer_data)
    
    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(customer_data_scaled)
    
    return df, kmeans

# Visualize customer segmentation
def plot_customer_segments(df):
    """Plots the customer segments based on clustering."""
    
    plt.figure(figsize=(10, 7))
    
    # Plot by cluster and feature
    plt.scatter(df['OrderCount'], df['OrderAmountHikeFromlastYear'], c=df['Cluster'], cmap='viridis', s=50)
    
    plt.title("Customer Segmentation (KMeans Clustering)")
    plt.xlabel("Order Count")
    plt.ylabel("Order Amount Hike from Last Year")
    plt.colorbar(label='Cluster')
    plt.show()

# Function to determine optimal number of clusters using the elbow method
def find_optimal_clusters(df, max_clusters=10):
    """Uses the elbow method to determine the optimal number of clusters."""
    
    features = ['OrderCount', 'OrderAmountHikeFromlastYear', 'HourSpendOnApp', 'SatisfactionScore']
    customer_data = df[features].dropna()
    
    # Scale the data
    scaler = StandardScaler()
    customer_data_scaled = scaler.fit_transform(customer_data)
    
    sse = []
    
    for k in range(1, max_clusters+1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(customer_data_scaled)
        sse.append(kmeans.inertia_)  # Sum of squared distances to closest cluster center
    
    # Plot the elbow curve
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, max_clusters+1), sse, marker='o')
    plt.title("Elbow Method for Optimal Clusters")
    plt.xlabel("Number of Clusters")
    plt.ylabel("SSE (Sum of Squared Errors)")
    plt.show()
    
    # Find the "elbow" point (the point with the maximum drop)
    optimal_clusters = sse.index(min(sse[1:])) + 1  # Select the cluster number with the most significant drop
    
    print(f"Optimal number of clusters: {optimal_clusters}")
    return optimal_clusters

