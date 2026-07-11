import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import os

# 1. Configuration
DATASET_PATH = 'finance_dataset.csv'
MODEL_OUTPUT_PATH = 'financial_health_model.pkl'

def load_data():
    """Loads the dataset or generates synthetic data if the file is missing."""
    if os.path.exists(DATASET_PATH):
        print(f"Loading dataset from '{DATASET_PATH}'...")
        df = pd.read_csv(DATASET_PATH)
    else:
        print(f"Warning: '{DATASET_PATH}' not found!")
        print("Generating synthetic data for demonstration purposes...")
        np.random.seed(42)
        n_samples = 5000
        income = np.random.randint(20000, 150000, n_samples)
        expense = income * np.random.uniform(0.2, 1.3, n_samples)
        
        # Generating synthetic category-wise expenses
        food_exp = expense * np.random.uniform(0.1, 0.5, n_samples)
        transport_exp = expense * np.random.uniform(0.05, 0.3, n_samples)
        entertainment_exp = expense * np.random.uniform(0.0, 0.4, n_samples)
        shopping_exp = expense * np.random.uniform(0.0, 0.4, n_samples)
        
        df = pd.DataFrame({
            'monthly_income': income,
            'monthly_expense_total': expense,
            'budget_goal': income * 0.8,
            'food_expense': food_exp,
            'transport_expense': transport_exp,
            'entertainment_expense': entertainment_exp,
            'shopping_expense': shopping_exp
        })
    return df

def preprocess_data(df):
    """Cleans and prepares features for the machine learning model."""
    print("Preprocessing data and performing feature engineering...")
    
    # Map the EXACT column names from the Kaggle dataset
    if 'monthly_income' in df.columns:
        df['total_income'] = df['monthly_income']
    else:
        df['total_income'] = 0 
        
    # Kaggle dataset uses 'monthly_expense_total'
    if 'monthly_expense_total' in df.columns:
        df['total_expense'] = df['monthly_expense_total']
    elif 'monthly_expense' in df.columns: 
        df['total_expense'] = df['monthly_expense']
    else:
        df['total_expense'] = 0 
        
    # Handle missing values by filling them with the median
    df['total_income'] = df['total_income'].fillna(df['total_income'].median())
    df['total_expense'] = df['total_expense'].fillna(df['total_expense'].median())
    
    # Feature 1: Savings Rate = (Income - Expense) / Income
    df['savings_rate'] = np.where(
        df['total_income'] > 0, 
        (df['total_income'] - df['total_expense']) / df['total_income'], 
        0
    )
    
    # Feature 2: Budget Utilization
    # Kaggle dataset uses 'budget_goal'
    if 'budget_goal' in df.columns:
        df['budget_limit'] = df['budget_goal'].fillna(df['total_income'] * 0.8)
    elif 'budget_limit' in df.columns:
        df['budget_limit'] = df['budget_limit'].fillna(df['total_income'] * 0.8)
    else:
        df['budget_limit'] = df['total_income'] * 0.8

    df['budget_utilization'] = np.where(
        df['budget_limit'] > 0, 
        df['total_expense'] / df['budget_limit'], 
        df['total_expense'] / df['total_income']
    )
    
    # NEW FEATURES: Category Ratios (This enables specific transaction insights)
    # If the dataset doesn't have these columns, create 0s
    for cat in ['food_expense', 'transport_expense', 'entertainment_expense', 'shopping_expense']:
        if cat not in df.columns:
            df[cat] = 0
            
    df['food_ratio'] = np.where(df['total_income'] > 0, df['food_expense'] / df['total_income'], 0)
    df['ent_ratio'] = np.where(df['total_income'] > 0, df['entertainment_expense'] / df['total_income'], 0)
    df['shop_ratio'] = np.where(df['total_income'] > 0, df['shopping_expense'] / df['total_income'], 0)
    
    # Replace any infinite values with 0 (in case of division by zero)
    df.replace([np.inf, -np.inf], 0, inplace=True)
    
    return df

def generate_labels(df):
    """Generates target labels based on strict financial rules for training."""
    print("Generating target labels for training...")
    
    def calculate_actionable_insight(row):
        # 0 = High Risk (General), 1 = Moderate, 2 = Healthy
        # NEW CLASSES: 3 = High Food Spend, 4 = High Entertainment Spend, 5 = High Shopping Spend
        if row['ent_ratio'] > 0.15: # Spending more than 15% of income on entertainment
            return 4
        elif row['shop_ratio'] > 0.20: # Spending more than 20% on shopping
            return 5
        elif row['food_ratio'] > 0.30: # Spending more than 30% on food
            return 3
        elif row['savings_rate'] < 0.05 or row['budget_utilization'] >= 0.95:
            return 0
        elif row['savings_rate'] >= 0.20 and row['budget_utilization'] <= 0.75:
            return 2
        else:
            return 1
            
    df['health_status'] = df.apply(calculate_actionable_insight, axis=1)
    return df

def train_model():
    """Trains the Random Forest model and saves it to a .pkl file."""
    df = load_data()
    df = preprocess_data(df)
    df = generate_labels(df)
    
    # Added the new category ratio features to the model's inputs
    features = ['total_income', 'total_expense', 'savings_rate', 'budget_utilization', 'food_ratio', 'ent_ratio', 'shop_ratio']
    X = df[features]
    y = df['health_status']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            class_weight='balanced',
            random_state=42
        ))
    ])
    
    print("Training the Actionable Random Forest model...")
    pipeline.fit(X_train, y_train)
    
    train_accuracy = pipeline.score(X_train, y_train)
    test_accuracy = pipeline.score(X_test, y_test)
    
    print("-" * 30)
    print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
    print(f"Testing Accuracy:  {test_accuracy * 100:.2f}%")
    print("-" * 30)
    
    if test_accuracy >= 0.90:
        print("Success! The model has achieved the target accuracy (>90%).")
    else:
        print("Notice: The accuracy is below 90%. You may need more data or cleaner dataset columns.")
        
    joblib.dump(pipeline, MODEL_OUTPUT_PATH)
    print(f"Model successfully saved as '{MODEL_OUTPUT_PATH}'.")
    print("You can now start your Flask app (python app.py) to use real-time transaction-level AI predictions!")

if __name__ == '__main__':
    train_model()