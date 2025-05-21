# Customer Churn Prediction

This project predicts customer churn using a Random Forest Classifier based on transaction data. It includes data preprocessing, model training, evaluation, and a retraining mechanism to ensure model performance. Built as part of [Your Course/Project Name] in May 2025.

## Features
- **Data Preprocessing**: Handles missing values using `dropna()` and selects features like transaction count, average transaction amount, days since last transaction, customer age, and account tenure.
- **Model**: Random Forest Classifier with 100 estimators and a max depth of 10 for robust predictions.
- **Evaluation**: Computes accuracy and confusion matrix to evaluate model performance.
- **Retraining**: Automatically retrains the model if accuracy falls below a threshold of 0.75.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/Churn-Prediction.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Prepare a `transaction_data.csv` file in the `data/` folder with the following columns:
   - `transaction_count`: Number of transactions
   - `avg_transaction_amount`: Average transaction amount
   - `days_since_last_transaction`: Days since the last transaction
   - `customer_age`: Age of the customer
   - `account_tenure`: Duration of account in days
   - `churn`: Target variable (0 for no churn, 1 for churn)

## Usage
1. Place your `transaction_data.csv` in the `data/` folder.
2. Run the script:
   ```bash
   python src/churn_prediction.py
   ```
3. The script will:
   - Load and preprocess the data
   - Train a Random Forest model
   - Save the model as `models/churn_model.pkl`
   - Check if retraining is needed and save the updated model as `models/churn_model_updated.pkl`

## Project Structure
- `src/churn_prediction.py`: Main script for loading, preprocessing, training, and retraining.
- `data/`: Placeholder for input data (not included in the repository, see `.gitignore`).
- `models/`: Stores trained models (not included in the repository, see `.gitignore`).
- `requirements.txt`: Lists required Python libraries.
- `.gitignore`: Excludes unnecessary files like `.pkl` and `.csv`.

## Example Output
```
Loading data...
Features shape: (1000, 5), Target shape: (1000,)
Training model...
Model Accuracy: 0.82
Confusion Matrix:
[[150  10]
 [ 20  70]]
Model saved to models/churn_model.pkl
Checking if retraining is needed...
Accuracy 0.82 is fine, no retraining needed.
Model saved to models/churn_model_updated.pkl
```

## Challenges Faced
- Handling missing data: Used `dropna()` for simplicity, but could explore imputation for better results.
- Model tuning: Limited `max_depth` to 10 to prevent overfitting while maintaining good accuracy.
- File paths: Used `os.path` for cross-platform compatibility.

## Future Improvements
- Add hyperparameter tuning using GridSearchCV.
- Implement cross-validation for more robust evaluation.
- Add unit tests to verify preprocessing and model training.

## License
MIT License (optional, included for open-source sharing).

---
Built by [Your Name] for [Your Course/Project Name], May 2025.