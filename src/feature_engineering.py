import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates business features for modeling.
    """
    logger.info("Engineering features...")
    
    # Basic Cost Features
    df['total_cost'] = df['Quantity'] * df['UnitPrice']
    df['cost_per_unit'] = df['UnitPrice']
    df['vendor_id'] = df['CustomerID']
    
    # Simulate freight cost: Base cost + distance/quantity factor + noise
    np.random.seed(42)
    df['freight_cost'] = 5.0 + (df['Quantity'] * 0.05) + np.random.normal(0, 1.5, len(df))
    df['freight_cost'] = df['freight_cost'].clip(lower=1.0) # Freight cannot be negative
    
    # Vendor-level aggregation features
    vendor_stats = df.groupby('vendor_id').agg(
        vendor_avg_cost=('total_cost', 'mean'),
        transaction_count=('InvoiceNo', 'nunique')
    ).reset_index()
    
    df = df.merge(vendor_stats, on='vendor_id', how='left')
    
    # Create Risk Flag (Classification Target)
    # Rule: If freight cost is unusually high compared to the total order cost (> 30%) 
    # and the order is substantial (>$20), flag as risky (1). Otherwise safe (0).
    df['risk_flag'] = np.where(
        (df['freight_cost'] > 0.30 * df['total_cost']) & (df['total_cost'] > 20.0), 
        1, 
        0
    )
    
    logger.info(f"Feature engineering complete. Risky invoices detected: {df['risk_flag'].sum()}")
    return df
