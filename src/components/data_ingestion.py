import pandas as pd
import os
from dataclasses import dataclass

from app_logger.logger import logging

@dataclass
class DataIngestionConfig:
    """
    Configuration class for data ingestion.
    
    Attributes:
    - file_path (str): Path to the CSV file to be ingested.
    """
    file_path: str = os.path.join("artifacts", "data")
    train_path: str = os.path.join("artifacts", "data", "train.csv")
    test_path: str = os.path.join("artifacts", "data", "test.csv")
    
    
    
class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()
    
    def load_data(self):
        """
        Load data from a CSV file.
        
        Returns:
        pd.DataFrame: Loaded data as a DataFrame.
        """
        try:
            train_df = pd.read_csv(self.config.train_path)
            # print(f"Train data loaded successfully from {self.config.train_path}")
            logging.info(f"Train data loaded successfully from {self.config.train_path}")
            
            test_df = pd.read_csv(self.config.test_path)
            # print(f"Test data loaded successfully from {self.config.test_path}")
            logging.info(f"Test data loaded successfully from {self.config.test_path}")
            
            print(f"Test data loaded successfully from {self.config.file_path}")
            
            return train_df, test_df
        except Exception as e:
            print(f"Error loading data: {e}")
            logging.error(f"Error loading data: {e}")
            return None