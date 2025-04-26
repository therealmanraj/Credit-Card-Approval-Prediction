import os
from pathlib import Path
import sys
from dataclasses import dataclass

import joblib
from sklearn.ensemble import GradientBoostingClassifier


from utils.common import CustomException
from utils.common import save_object

from app_logger.logger import logging

random_state = 42


@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts", "models", "model.pkl")
    trained_model_dir_path=os.path.join("artifacts", "models")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train):
        try:
            logging.info("Splitting training data")
            X_train,y_train = train.loc[:,train.columns != 'Is high risk'], train['Is high risk'].astype('int64')
            
            models = {
                'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            }
            
            os.makedirs(self.model_trainer_config.trained_model_dir_path, exist_ok=True)
            model_trn = models['Gradient Boosting'].fit(X_train, y_train)
            joblib.dump(model_trn, self.model_trainer_config.trained_model_file_path)
            
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_model_prediction(self,test):
        try:
            logging.info("Test input data")
            X_test,y_test = test.loc[:,test.columns != 'Is high risk'], test['Is high risk'].astype('int64')
            
            model_trn = joblib.load(self.model_trainer_config.trained_model_file_path)
            
            final_prediction = model_trn.predict(X_test)
            
            return final_prediction
            
        except Exception as e:
            raise CustomException(e,sys)
        
    def predict(self,predict_df):
        try:
            logging.info("Test input data")
            
            model_trn = joblib.load(self.model_trainer_config.trained_model_file_path)
            
            predict_df = predict_df.drop(columns='Is high risk', errors='ignore')
            final_prediction = model_trn.predict(predict_df)
            
            return final_prediction
            
        except Exception as e:
            raise CustomException(e,sys)