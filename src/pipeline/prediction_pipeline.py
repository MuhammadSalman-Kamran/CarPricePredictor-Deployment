from src.logger import logging
from src.exception import CustomException
import sys
import os
import pandas as pd
from src.utils import load_obj

class Prediction:
    def __init__(self) -> None:
        pass

    def prediction(self, input):
        logging.info('Into Prediction Function')
        try:
            logging.info('Defining the paths of the model and preprocessor object')
            model_file_path = os.path.join('artifacts', 'model.pkl')
            preprocessor_file_path = os.path.join('artifacts', 'preprocessor.pkl')

            logging.info('Loading the model and preprocessor object')
            model = load_obj(model_file_path)
            preprocessor = load_obj(preprocessor_file_path)
            logging.info('Transforming the User input')
            processed_input = preprocessor.transform(input)
            logging.info('Making the prediction using processed input')
            prediction = model.predict(processed_input)

            return prediction
        except Exception as e:
            raise CustomException(e, sys)
        
class CustomData:
    def __init__(self,selected_model,company, year, kilo , type):
        self.selected_model = selected_model
        self.company = company
        self.year = year
        self.kilo = kilo
        self.type = type

    def data_as_df(self):
        return (pd.DataFrame([[self.selected_model,self.company, self.year, self.kilo , self.type]], columns = ['name', 'company', 'year', 'kms_driven' ,'fuel_type']))