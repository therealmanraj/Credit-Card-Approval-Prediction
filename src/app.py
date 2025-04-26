from components.data_ingestion import DataIngestion, DataIngestionConfig
from components.data_pipeline import DataTransformationConfig, DataPipeline
from components.model_training import ModelTrainer, ModelTrainerConfig

from app_logger.logger import logging

import pandas as pd
pd.set_option('display.max_columns', None)

# Before running change the directory to the src folder
DataIngestionConfig = DataIngestionConfig()
data_ingest = DataIngestion()
train, test = data_ingest.load_data()
print("Data Ingestion done successfully.")
predict = train.tail(10)

DataTransformationConfig = DataTransformationConfig()
data_pipeline = DataPipeline()
train_df, test_df = data_pipeline.data_transformation(train, test)
print("Data Transformation done successfully.")

ModelTrainerConfig = ModelTrainerConfig()
model_training = ModelTrainer()
model_training.initiate_model_trainer(train_df)
print("Model Training done successfully.")

accuracy, pred = model_training.initiate_model_prediction(test_df)
print("Model Prediction done successfully.")
print(f"Accuracy of the model is: {accuracy}")
print(f"Predictions of the model is: {pred}")