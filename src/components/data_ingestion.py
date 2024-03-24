import sys
import os
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from zenml import step
from src.utils import train_test_split_csv

@dataclass
class DataIngestionConfig:
    sample_data_file_path = os.path.join('artifacts','sample.csv')
    train_data_file_path = os.path.join('artifacts', 'train.csv')
    test_data_file_path = os.path.join('artifacts', 'test.csv')

class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def ingestion(self):
        logging.info('Ingestion of the data has started')
        try:
            logging.info('Reading the csv data from file path')
            df = pd.read_csv('notebook/data/clean_data.csv')
            df = df.iloc[:, 1:]
            logging.info('Making the directory for storing csv files')
            os.makedirs(os.path.dirname(self.data_ingestion_config.sample_data_file_path), exist_ok=True)
            logging.info('Storing the Sample Data')
            df.to_csv(self.data_ingestion_config.sample_data_file_path, index=False, header=True)

            train_df, test_df = train_test_split_csv(df, 0.2)

            logging.info('Saving Training CSV file')
            train_df.to_csv(self.data_ingestion_config.train_data_file_path, index = False, header = True)
            
            logging.info('Saving Testing CSV file')
            test_df.to_csv(self.data_ingestion_config.test_data_file_path, index = False, header = True)

            return (
                self.data_ingestion_config.train_data_file_path,
                self.data_ingestion_config.test_data_file_path
            )
        
        except Exception as e:
            raise CustomException(e, sys)