import sys
import os
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from src.utils import save_obj


@dataclass
class DataTransformationConfig:
    preprocessor_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self) -> None:
        self.data_transformation_config = DataTransformationConfig()

    def make_pipeline(self):
        logging.info('Process of making Scikit-Learn Pipeline')
        try:
            numerical_col = ['year','kms_driven']
            categorical_col = ['name','company','fuel_type']

            logging.info('Making Pipeline for numerical Columns')
            numerical_pipe = Pipeline([
                ('Imputing', SimpleImputer(strategy='mean')),
                ('Scaling', StandardScaler())
            ])

            logging.info('Making pipeline for categorical columns')
            categorical_pipe = Pipeline([
                ('Imputing', SimpleImputer(strategy='most_frequent')),
                ('Encoding', OneHotEncoder(handle_unknown='ignore'))
            ])

            logging.info('Combining both pipelines using ColumnTransformer')
            preprocessor = ColumnTransformer([
                ('Numerical Pipeline', numerical_pipe, numerical_col),
                ('Categorical Pipe', categorical_pipe, categorical_col)
            ])

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def transformation(self, train_file_path, test_file_path):
        logging.info('Transformation process has started')
        try:
            logging.info('Reading Training and testing file in their DataFrames')
            train_df = pd.read_csv(train_file_path)
            test_df = pd.read_csv(test_file_path)
            
            logging.info('Making the Preprocessor obj')
            preprocessor_obj = self.make_pipeline()

            logging.info('Dividing the Independant and Dependant Columns')
            train_input_before_process = train_df.drop(['Price'], axis = 1)
            train_output = train_df['Price']
            test_input_before_process = test_df.drop(['Price'], axis = 1)
            test_output = test_df['Price']

            logging.info('Processing the input of training and testing file')
            train_processed_input = preprocessor_obj.fit_transform(train_input_before_process)
            test_processed_input = preprocessor_obj.transform(test_input_before_process)
            # train_output = np.array(train_output).reshape(-1, 1)

            # print(train_processed_input.shape)  # Should be (653, ...)
            # print(np.array(train_output).shape)  # Should be (653, 1)

            # logging.info('Concatenating the input and output into array')
            # train_array = np.c_[train_processed_input, train_output]
            # test_array = np.c_[test_processed_input, np.array(test_output)]

            save_obj(self.data_transformation_config.preprocessor_file_path, preprocessor_obj)

            return (train_processed_input, test_processed_input)
        except Exception as e:
            raise CustomException(e, sys)
        