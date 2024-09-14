from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
import numpy as np

from data_cleaning import encode_categorical

# Set a random seed for reproducibility
def set_seed(seed=42):
    """Set seed for reproducibility."""
    tf.random.set_seed(seed)
    np.random.seed(seed)

# Function to split the data into training and testing sets
def preprocess_data(df, random_state = 42):
    """Prepares the data for modeling by encoding and splitting."""
    
    # Encoding categorical features (this function comes from data_cleaning.py)
    df = encode_categorical(df)
    
    # Separating features (X) and target variable (y)
    X = df.drop(['CustomerID', 'Churn'], axis=1)
    y = df['Churn']
    
    # Splitting the data into training and test sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)

    # Initialize the StandardScaler
    scaler = StandardScaler()
    
    # Fit the scaler on the training data and transform both training and test data
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
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

# Function to build a TensorFlow Neural Network
def build_nn_model(input_shape):
    """Builds a simple Neural Network with TensorFlow/Keras."""
    
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(64, activation='relu', input_shape=(input_shape,)))
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))  # Sigmoid for binary classification
    
    model.compile(optimizer='adam', 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    
    return model

def train_nn_model(X_train, X_test, y_train, y_test):
    """Trains and evaluates a neural network model using TensorFlow."""

    # Set the random seed
    set_seed()
    
    # Build the model
    model = build_nn_model(X_train.shape[1])
    
    # Train the model
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)
    
    # Predict probabilities
    y_pred_prob = model.predict(X_test).ravel()
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    # Evaluate the model using sklearn metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_prob)
    
    performance = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'AUC-ROC': roc_auc
    }
    
    return performance
