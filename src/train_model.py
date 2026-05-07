import os
import pickle
import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, classification_report, accuracy_score

logger = logging.getLogger(__name__)

def train_and_evaluate(df: pd.DataFrame, models_dir: str):
    """
    Trains regression and classification models, compares them, and saves the best to disk.
    """
    os.makedirs(models_dir, exist_ok=True)
    
    # --- Regression: Predict Freight Cost ---
    logger.info("Preparing data for Regression (Freight Cost)...")
    reg_features = ['Quantity', 'UnitPrice', 'total_cost', 'vendor_avg_cost', 'transaction_count']
    X_reg = df[reg_features]
    y_reg = df['freight_cost']
    
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
    
    regressors = {
        'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        'DecisionTree': DecisionTreeRegressor(max_depth=10, random_state=42),
        'LinearRegression': LinearRegression()
    }
    
    best_reg_name = None
    best_reg_model = None
    best_mae = float('inf')
    
    for name, model in regressors.items():
        logger.info(f"Training {name} Regressor...")
        model.fit(Xr_train, yr_train)
        preds = model.predict(Xr_test)
        mae = mean_absolute_error(yr_test, preds)
        logger.info(f"{name} Regressor MAE: {mae:.4f}")
        
        if mae < best_mae:
            best_mae = mae
            best_reg_model = model
            best_reg_name = name
            
    logger.info(f"Best Regressor is {best_reg_name} with MAE: {best_mae:.4f}")
    with open(os.path.join(models_dir, 'freight_regressor.pkl'), 'wb') as f:
        pickle.dump(best_reg_model, f)
    
    # --- Classification: Detect Risky Invoices ---
    logger.info("Preparing data for Classification (Risk Flag)...")
    clf_features = ['Quantity', 'UnitPrice', 'total_cost', 'freight_cost', 'vendor_avg_cost', 'transaction_count']
    X_clf = df[clf_features]
    y_clf = df['risk_flag']
    
    Xc_train, Xc_test, yc_train, yc_test = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)
    
    classifiers = {
        'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1),
        'DecisionTree': DecisionTreeClassifier(max_depth=10, class_weight='balanced', random_state=42),
        'LogisticRegression': LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
    }
    
    best_clf_name = None
    best_clf_model = None
    best_accuracy = -1
    
    for name, model in classifiers.items():
        logger.info(f"Training {name} Classifier...")
        model.fit(Xc_train, yc_train)
        preds = model.predict(Xc_test)
        acc = accuracy_score(yc_test, preds)
        logger.info(f"{name} Classifier Accuracy: {acc:.4f}")
        
        if acc > best_accuracy:
            best_accuracy = acc
            best_clf_model = model
            best_clf_name = name
            
    logger.info(f"Best Classifier is {best_clf_name} with Accuracy: {best_accuracy:.4f}")
    
    best_preds = best_clf_model.predict(Xc_test)
    logger.info(f"\nClassification Report for {best_clf_name}:\n" + classification_report(yc_test, best_preds))
    
    with open(os.path.join(models_dir, 'risk_classifier.pkl'), 'wb') as f:
        pickle.dump(best_clf_model, f)
    logger.info(f"Models saved successfully to {models_dir}")
