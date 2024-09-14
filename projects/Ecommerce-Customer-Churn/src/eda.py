import matplotlib.pyplot as plt
import seaborn as sns

# Function for EDA (Improved)
def eda(df):
    """Performs an extended EDA including correlation, histograms, and boxplots."""
    
    # Plot histograms for numerical features
    df.hist(bins=30, figsize=(15, 10))
    plt.suptitle("Histograms of Numerical Features", size=16)
    plt.show()

    # Correlation heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title("Correlation Heatmap")
    plt.show()

    # Boxplot to check for outliers
    plt.figure(figsize=(15, 10))
    sns.boxplot(data=df[['Tenure', 'WarehouseToHome', 'HourSpendOnApp', 'OrderAmountHikeFromlastYear', 'OrderCount', 'DaySinceLastOrder']])
    plt.title("Boxplots of Numerical Features (Checking for Outliers)")
    plt.show()
