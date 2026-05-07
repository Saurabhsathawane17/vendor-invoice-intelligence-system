import os
import shap
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

def load_models(models_dir):
    with open(os.path.join(models_dir, "freight_regressor.pkl"), "rb") as f:
        reg_model = pickle.load(f)
    with open(os.path.join(models_dir, "risk_classifier.pkl"), "rb") as f:
        clf_model = pickle.load(f)
    return reg_model, clf_model


def predict_invoice_risk(data, models_dir):
    reg_model, clf_model = load_models(models_dir)

    df = pd.DataFrame([data])

    df['total_cost'] = df['Quantity'] * df['UnitPrice']
    df['vendor_avg_cost'] = data.get('vendor_avg_cost', df['total_cost'] * 0.05)
    df['transaction_count'] = data.get('transaction_count', 10)

    # --- 1. Predict Freight Cost (Regression) ---
    reg_features = ['Quantity', 'UnitPrice', 'total_cost', 'vendor_avg_cost', 'transaction_count']
    X_reg = df[reg_features]

    expected_freight = reg_model.predict(X_reg)[0]
    
    provided_freight = data.get('freight_cost')
    if not provided_freight or provided_freight <= 0:
        provided_freight = expected_freight

    # --- 2. Predict Risk (Classification) ---
    df['freight_cost'] = provided_freight
    clf_features = ['Quantity', 'UnitPrice', 'total_cost', 'freight_cost', 'vendor_avg_cost', 'transaction_count']
    X_clf = df[clf_features]

    risk_flag = clf_model.predict(X_clf)[0]

    # 🔥 SHAP Explainability
    shap_values = None
    try:
        # Bypass Python 3.12 serialization bug by calling TreeExplainer explicitly
        if hasattr(clf_model, 'tree_') or hasattr(clf_model, 'estimators_'):
            explainer = shap.TreeExplainer(clf_model)
        else:
            explainer = shap.Explainer(clf_model, X_clf)
            
        shap_raw = explainer(X_clf)
        # If binary classification, isolate the target class (Risk = 1)
        shap_values = shap_raw[:, :, 1] if len(shap_raw.shape) == 3 else shap_raw
    except Exception as e:
        print(f"SHAP explanation bypassed due to internal error: {e}")

    return {
        "expected_freight_cost": expected_freight,
        "provided_freight_cost": provided_freight,
        "risk_flag": risk_flag,
        "shap_values": shap_values,
        "features": X_clf.iloc[0].to_dict()
    }