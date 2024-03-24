import sys
import os
import numpy as np
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from src.utils import evaluate_model
from src.utils import save_obj

@dataclass
class ModelTrainingConfig:
    model_file_path = os.path.join('artifacts','model.pkl')

class ModelTraining:
    def __init__(self) -> None:
        self.model_training_config = ModelTrainingConfig()

    def training(self, input_train_arr, input_test_arr,train_file_path, test_file_path):
        logging.info('Training Process has started')
        try:
            logging.info('Training and Testing file imported successfully')
            train_df = pd.read_csv(train_file_path)
            test_df = pd.read_csv(test_file_path)

            logging.info('Converting output into numpy array')
            train_output = train_df['Price']
            train_output = np.array(train_output)
            test_output = test_df['Price']
            test_output = np.array(test_output)
            
            logging.info('Calling the model')
            model = LinearRegression()
            logging.info('Calling the evalution model')
            r_score, mean_score = evaluate_model(model, input_train_arr,train_output, input_test_arr, test_output)
            # print(r_score)
            # print(mean_score)

            logging.info('Dumping the model obj')
            save_obj(self.model_training_config.model_file_path, model)

            logging.info('Returning the model object folder path')
            return self.model_training_config.model_file_path
        except Exception as e:
            raise CustomException(e, sys)
