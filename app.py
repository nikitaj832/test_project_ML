import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Generate synthetic data
def generate_data():
    np.random.seed(42)
    X = np.random.rand(100, 1) * 10  # Feature: random values between 0 and 10
    y = 3 * X.squeeze() + np.random.randn(100) * 2  # Target: y = 3x + noise
    return X, y

# Train and evaluate model
def train_and_evaluate():
    # Generate data
    X, y = generate_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate R² score
    score = r2_score(y_test, y_pred)
    return score

if __name__ == "__main__":
    r2 = train_and_evaluate()
    print(f"R² Score: {r2}")
