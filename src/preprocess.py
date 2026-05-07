import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_preprocess(filepath: str) -> pd.DataFrame:
    """
    Loads raw CSV data and performs initial cleaning.
    """
    logger.info(f"Loading data from {filepath}...")
    try:
        df = pd.read_csv(filepath, encoding='ISO-8859-1')
    except FileNotFoundError:
        logger.error(f"File {filepath} not found. Please ensure the dataset exists.")
        raise
        
    initial_shape = df.shape
    
    # Remove null values for crucial columns
    df = df.dropna(subset=['CustomerID', 'Description'])
    
    # Remove negative or zero Quantity and UnitPrice
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
    
    # Convert data types properly
    df['CustomerID'] = df['CustomerID'].astype(int).astype(str)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    logger.info(f"Data cleaned. Shape went from {initial_shape} to {df.shape}")
    
    return df
