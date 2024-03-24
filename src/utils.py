import pandas as pd
import os
import sys
from sklearn.model_selection import train_test_split
import pickle
from src.logger import logging
from src.exception import CustomException
from sklearn.metrics import r2_score, mean_absolute_error

def train_test_split_csv(data, test_size):
    train_df, test_df = train_test_split(data, test_size=test_size, random_state=42)
    return (train_df, test_df)

def save_obj(preprocess_file_path, obj):
    try:
        logging.info('Making Directory for storing the object')
        os.makedirs(os.path.dirname(preprocess_file_path), exist_ok=True)
        logging.info('Saving the object ')
        pickle.dump(obj, open(preprocess_file_path, 'wb'))
    except Exception as e:
        raise CustomException(e, sys)
    
def load_obj(file_path):
    try:
        logging.info('Loading the object')
        return pickle.load(open(file_path, 'rb'))
    
    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_model(model, train_input, train_output, test_input, test_output):
    try:
        logging.info('Model Training is going to start')
        model.fit(train_input, train_output)
        prediction = model.predict(test_input)
        logging.info('Checking the error rate of the model')
        # error_score = mean_absolute_error(test_output, prediction)
        r_score = r2_score(test_output, prediction)
        absolute_score = mean_absolute_error(test_output, prediction)

        return (r_score, absolute_score)
    
    except Exception as e:
        raise CustomException(e, sys)