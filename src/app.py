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
# print(predict)

DataTransformationConfig = DataTransformationConfig()
data_pipeline = DataPipeline()
train_df, test_df = data_pipeline.data_transformation(train, test)
print("Data Transformation done successfully.")

ModelTrainerConfig = ModelTrainerConfig()
model_training = ModelTrainer()
model_training.initiate_model_trainer(train_df)
print("Model Training done successfully.")

pred = model_training.initiate_model_prediction(test_df)
print("Model Prediction done successfully.")

# print(f"Predict data: {predict.head()}")
transformed = data_pipeline.transformation(predict)
print("Predict Data Transformation done successfully.")
# print(f"Transformed data: {transformed.head()}")

pred = model_training.predict(transformed)
print("Model Prediction done successfully.")
print(f"Predicted data: {pred}")