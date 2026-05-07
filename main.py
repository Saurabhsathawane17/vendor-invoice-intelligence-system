import os
import logging
from src.preprocess import load_and_preprocess
from src.feature_engineering import engineer_features
from src.train_model import train_and_evaluate

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def main():
    data_path = 'data/online_retail.csv'
    models_dir = 'models/'
    
    # Ensure directories exist
    os.makedirs('data', exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    if not os.path.exists(data_path):
        logging.error(f"Dataset not found at {data_path}. Please place 'online_retail.csv' in the 'data/' folder.")
        return
        
    # 1. Preprocess
    df = load_and_preprocess(data_path)
    
    # 2. Engineer Features
    df = engineer_features(df)
    
    # 3. Train and Evaluate Models
    train_and_evaluate(df, models_dir)
    
    logging.info("End-to-end pipeline completed successfully!")

if __name__ == "__main__":
    main()
