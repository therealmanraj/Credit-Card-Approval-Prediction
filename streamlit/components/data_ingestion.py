import pandas as pd
import os
from dataclasses import dataclass

# from streamlit.logging.logger import logging

@dataclass
class DataIngestionConfig:
    """
    Configuration class for data ingestion.
    
    Attributes:
    - file_path (str): Path to the CSV file to be ingested.
    """
    train_path: str = os.path.join("artifacts", "data", "train.csv")
    test_path: str = os.path.join("artifacts", "data", "test.csv")
    
    
    
class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()
    
    def load_data(self):
        """
        Load data from a CSV file.
        
        Parameters:
        file_path (str): Path to the CSV file.
        
        Returns:
        pd.DataFrame: Loaded data as a DataFrame.
        """
        try:
            data = pd.read_csv(self.config.train_path)
            # logging.info(f"Data loaded successfully from {self.config.train_path}")
            return data
        except Exception as e:
            print(f"Error loading data: {e}")
            # logging.error(f"Error loading data: {e}")
            return None