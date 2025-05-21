# churn_prediction.py
# Script to predict customer churn using random forest
# Built for transaction data, last updated May 2025

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

# Load the dataset
def load_data(file_path):
    print("Loading data...")
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found. Please ensure the file exists.")
        return None

# Preprocess the data
def preprocess_data(df):
    if df is None:
        print("Error: No data to preprocess.")
        return None, None
    # Drop any rows with missing values (simple approach)
    df = df.dropna()
    
    # Define features and target
    features = ['transaction_count', 'avg_transaction_amount', 'days_since_last_transaction', 'customer_age', 'account_tenure']
    X = df[features]
    y = df['churn']
    
    # Check if data looks okay
    print(f"Features shape: {X.shape}, Target shape: {y.shape}")
    return X, y

# Train the random forest model
def train_model(X, y):
    if X is None or y is None:
        print("Error: Cannot train model with no data.")
        return None, None, None
    # Split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Initialize model
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    
    # Train model
    print("Training model...")
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"Model Accuracy: {accuracy:.2f}")
    print("Confusion Matrix:")
    print(cm)
    
    return model, X_test, y_test

# Save the model
def save_model(model, filename):
    if model is None:
        print("Error: No model to save.")
        return
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")

# Retrain model if accuracy drops
def retrain_model(model, X, y, threshold=0.75):
    if model is None or X is None or y is None:
        print("Error: Cannot retrain with invalid model or data.")
        return None
    # Simple retraining logic
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    if accuracy < threshold:
        print(f"Accuracy {accuracy:.2f} below threshold, retraining...")
        model.fit(X_train, y_train)
        y_pred_new = model.predict(X_test)
        new_accuracy = accuracy_score(y_test, y_pred_new)
        print(f"New accuracy after retraining: {new_accuracy:.2f}")
    else:
        print(f"Accuracy {accuracy:.2f} is fine, no retraining needed.")
    
    return model

# Main function to run everything
def main():
    # File path (assuming CSV file)
    file_path = r'E:\Tech\DST\Old Code\Churn-Prediction\transaction_data.csv'
    
    # Load and preprocess
    data = load_data(file_path)
    X, y = preprocess_data(data)
    
    # Train model
    model, X_test, y_test = train_model(X, y)
    
    # Save model
    save_model(model, 'churn_model.pkl')
    
    # Simulate monitoring and retraining
    print("\nChecking if retraining is needed...")
    model = retrain_model(model, X, y)
    
    # Save updated model
    save_model(model, 'churn_model_updated.pkl')

if __name__ == "__main__":
    main()