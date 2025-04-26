from components.data_ingestion import DataIngestion, DataIngestionConfig
from components.data_pipeline import DataTransformationConfig, DataPipeline

from app_logger.logger import logging

# Before running change the directory to the src folder
DataIngestionConfig = DataIngestionConfig()
data_ingest = DataIngestion()
train, test = data_ingest.load_data()
print("Data Ingestion done successfully.")

logging("Train data shape original:", train.shape)
logging("Test data shape original:", test.shape)

DataTransformationConfig = DataTransformationConfig()
data_pipeline = DataPipeline()
train_array, test_array, _ = data_pipeline.data_transformation(train, test)
print("Data Transformation done successfully.")

logging("Train data shape:", train_array.shape)
logging("Test data shape:", test_array.shape)

