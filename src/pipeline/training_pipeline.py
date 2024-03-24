from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_training import ModelTraining

if __name__ == '__main__':
    ingestion_obj = DataIngestion()
    train_df_file_path, test_df_file_path = ingestion_obj.ingestion()

    transformation_obj = DataTransformation()
    train_arr, test_arr = transformation_obj.transformation(train_df_file_path, test_df_file_path)

    training_obj = ModelTraining()
    model_file_path = training_obj.training(train_arr, test_arr, train_df_file_path, test_df_file_path)