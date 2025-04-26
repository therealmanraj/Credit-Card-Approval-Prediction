import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import GradientBoostingClassifier


from utils.common import CustomException
from utils.common import save_object
from utils.common import load_object

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
            save_object(self.model_trainer_config.trained_model_file_path, model_trn)
            
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_model_prediction(self,test):
        try:
            logging.info("Test input data")
            X_test,y_test = test.loc[:,test.columns != 'Is high risk'], test['Is high risk'].astype('int64')
            
            model_trn = load_object(self.model_trainer_config.trained_model_file_path)
            
            final_prediction = model_trn.predict(X_test)
            
            accuracy = model_trn.score(X_test, y_test)
            
            return accuracy, final_prediction
            
        except Exception as e:
            raise CustomException(e,sys)