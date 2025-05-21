# generate_dataset.py
# Script to create a sample dataset for churn prediction
# Saves data to transaction_data.csv

import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Number of rows
n_samples = 1000

# Generate random data
data = {
    'transaction_count': np.random.randint(1, 100, n_samples),
    'avg_transaction_amount': np.random.uniform(10, 1000, n_samples).round(2),
    'days_since_last_transaction': np.random.randint(1, 365, n_samples),
    'customer_age': np.random.randint(18, 80, n_samples),
    'account_tenure': np.random.randint(1, 120, n_samples),
}

# Create churn column (simple logic: more days since last transaction = higher chance of churn)
data['churn'] = np.where(data['days_since_last_transaction'] > 180, 
                         np.random.choice([0, 1], n_samples, p=[0.3, 0.7]), 
                         np.random.choice([0, 1], n_samples, p=[0.8, 0.2]))

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
output_path = r'E:\Tech\DST\Old Code\Churn-Prediction\transaction_data.csv'
df.to_csv(output_path, index=False)
print(f"Dataset saved to {output_path}")

# Check first few rows
print("First 5 rows of dataset:")
print(df.head())