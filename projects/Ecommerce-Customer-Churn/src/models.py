from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

# Function to split the data into training and testing sets
def preprocess_data(df):
    """Prepares the data for modeling by encoding and splitting."""
    
    # Encoding categorical features (this function comes from data_cleaning.py)
    df = encode_categorical(df)
    
    # Separating features (X) and target variable (y)
    X = df.drop(['CustomerID', 'Churn'], axis=1)
    y = df['Churn']
    
    # Splitting the data into training and test sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    return X_train, X_test, y_train, y_test

# Function to train and evaluate multiple models
def train_multiple_models(X_train, X_test, y_train, y_test):
    """Trains and evaluates multiple models."""
    
    # Initialize models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }
    
    # Dictionary to store the performance of each model
    performance = {}

    for model_name, model in models.items():
        # Train the model
        model.fit(X_train, y_train)
        
        # Predict on the test set
        y_pred = model.predict(X_test)
        
        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        
        performance[model_name] = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'AUC-ROC': roc_auc
        }
        
    # Return the performance of all models
    return performance
